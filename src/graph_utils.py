import torch
import tqdm
import torch_geometric as tg
import pyarrow
import numpy as np

class graph_constructor: 

  def __init__(self,DF, max__iter_hubs, M_func):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.DF = DF.to(self.device)
    self.N = DF.size(dim=0)
    self.max__iter_hubs = max__iter_hubs
    self.M_func = M_func

  
  '''Private Methods'''

  def _M_from_index(self,Li):
    LM = []
    for i in Li:
      LM.append(self.M_func(self.DF[i,:]))
    return torch.stack(LM,dim=0).to(self.device)
  
  def _V_from_A(self):
    V = [[] for l in range(torch.max(self.H)+1)]
    for i in range(self.A.size(dim=0)):
      V[self.A[i].item()].append(i)
    return V
  
  def _allocation(self):
    diff = self.DF[self.H].unsqueeze(0) - self.DF.unsqueeze(1)
    distances = torch.einsum('nka,kab,nkb->nk', diff, self.M_hub, diff)
    return self.H[torch.argmin(distances, dim=1).cpu()],torch.argmin(distances, dim=1).cpu()

  def _criteria(self,iA,iB,iC):

    v_stack = torch.stack([
      self.DF[iA, :] - self.DF[iB, :],
      self.DF[iB, :] - self.DF[iC, :],
      self.DF[iA, :] - self.DF[iC, :],
      self.DF[iC, :] - self.DF[iA, :]
    ]).to(self.device)
    m_stack = torch.stack([
      self.M_hub[self.Aindex[iB], :, :],
      self.M_hub[self.Aindex[iB], :, :],
      self.M_hub[self.Aindex[iA], :, :],
      self.M_hub[self.Aindex[iC], :, :]
    ]).to(self.device)
    distances = torch.einsum('bn, bnm, bm -> b', v_stack, m_stack, v_stack)
    return (distances[0]+distances[1])/(distances[2]+distances[3])

  def _choose_hub(self):
    Lkmax_pair = []
    LRkmax_pair = []

    print('Loop 1: ',len(self.LP))

    for i_pair in range(len(self.LP)):
      pair = self.LP[i_pair]
      candidats = torch.cat((torch.tensor(self.V[pair[0]]),torch.tensor(self.V[pair[1]])),0)
      LRk = []

      print('Loop 2: ',candidats.size(dim=0))

      for k in candidats:
        Rk = 0

        print('Loop 3: ',len(self.V[pair[0]]))
        print('Loop 4: ',len(self.V[pair[1]]))

        for n in self.V[pair[0]]:
          for m in self.V[pair[1]]:
            Rk += self._criteria(n,k,m)
        LRk.append(Rk)
      Lkmax_pair.append(torch.argmax(LRk,dim=0))
      LRkmax_pair.append(LRk[Lkmax_pair[-1]])
    return Lkmax_pair[torch.argmax(LRkmax_pair,dim=0)]
  
  def _new_hub(self):
    k = self._choose_hub()
    self.H = torch.cat((self.H, k), 0)
    self.M_hub = torch.cat((self.M_hub, self.M_from_index([k]).unsqeeze(0)), 0).to(self.device)
    self.A = self._allocation()
    self.V = self.V_from_A()

  def _iter(self):
    old_A = self.A
    self._new_hub()
    LP = []
    for k in range(self.N):
      if old_A[k] != self.A[k]:
        LP.append((old_A[k],self.A[k]))
    self.LP = list(set(LP))

  def _distance_pnp(self,theta1,theta2,M):
    #'...' in einsum to handle both batched and single inputs
    v = theta1 - theta2
    return torch.einsum('...i,...ij,...j->...', v, M, v) 

  def _compute_row(self,theta_c,M_c,theta_r,M_r):
    # M_r (single matrix) and M_c (batched matrix)
    d1 = self._distance_pnp(theta_c, theta_r, M_r)
    d2 = self._distance_pnp(theta_r, theta_c, M_c)
    return (d1 + d2) / 2

  
  '''Public Methods'''
  
  def initialize(self):
    #Initialize with 2 random hubs
    self.H = torch.randint(0,self.N,(2,1)).squeeze(1) 
    self.M_hub = self._M_from_index([self.H[0].item(),self.H[1].item()])
    self.A,self.Aindex = self._allocation()
    self.V = self._V_from_A()
    self.LP = [(self.H[0].item(),self.H[1].item())]

  def S_hub_from_dataset(self, ds_path):
    id_hub_list = self.H.tolist()
    parquet_file = pyarrow.parquet.ParquetFile(ds_path)
    S_hub = []
    for i in tqdm.tqdm(range(parquet_file.num_row_groups), desc="Reading S(hubs)..."):
      table = parquet_file.read_row_group(i)
      mask = pyarrow.parquet.is_in(table["row_id"], pyarrow.array(id_hub_list))
      filtered_table = table.filter(mask)
      S_batch_cpu = torch.from_numpy(np.array(filtered_table.drop(["row_id"])))
      S_hub.append(S_batch_cpu)
    self.S_hub = torch.cat(S_hub)
    return self.S_hub 

  def construct_hubs(self):
    for i in tqdm.tqdm(range(self.max__iter_hubs),desc="Constructing hubs..."):
      print('H: ',self.H)
      print('LP: ',self.LP)
      self._iter()

  def construct_edges(self, k, batch_size=128):
    M_all = self.M_hub[self.A].to(self.device)  
    sources_list = []
    targets_list = []
    weights_list = []
    for i in tqdm.tqdm(range(0, self.N, batch_size), desc='Constructing edges...'):
      start = i
      end = min(i + batch_size, self.N)          
      batch_theta = self.DF[start:end]   
      batch_M = M_all[start:end]     
      dists_batch = []
      for b in range(end - start):
        d_row = self._compute_row(self.DF, M_all, batch_theta[b], batch_M[b]).cpu()
        dists_batch.append(d_row)
      dists_batch = torch.stack(dists_batch)
      vals, cols = torch.topk(dists_batch, k=k+1, dim=1, largest=False)
      rows = torch.arange(start, end, device=self.device).unsqueeze(1).repeat(1, k+1).cpu()
      mask = rows != cols
      valid_rows = rows[mask]
      valid_cols = cols[mask]
      valid_vals = vals[mask]
      # Both ways to get a symmetric graph
      sources_list.append(valid_rows)
      targets_list.append(valid_cols)
      weights_list.append(valid_vals)
      sources_list.append(valid_cols)
      targets_list.append(valid_rows)
      weights_list.append(valid_vals)
    all_sources = torch.cat(sources_list)
    all_targets = torch.cat(targets_list)
    all_weights = torch.cat(weights_list)
    edge_index = torch.stack([all_sources, all_targets], dim=0)
    self.edge_index = edge_index
    self.edge_attr = all_weights
    
  def construct_graph(self):
    self.graph = tg.data.Data(x=self.DF, edge_index=self.edge_index, edge_attr=self.edge_attr)

  def save_to_graphml(self, path):   
    edge_index = self.graph.edge_index.cpu().numpy()
    num_nodes = self.graph.num_nodes
    edge_weights = self.graph.edge_attr.cpu().numpy() if hasattr(self.graph, 'edge_attr') else None
    node_features = self.graph.x.cpu().numpy() if hasattr(self.graph, 'x') else None
    with open(path, 'w') as f:
      f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
      f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns" \n')
      f.write('         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" \n')
      f.write('         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')
      f.write('  <key id="v_feat" for="node" attr.name="features" attr.type="string"/>\n')
      f.write('  <key id="e_weight" for="edge" attr.name="weight" attr.type="double"/>\n')
      f.write('  <graph id="G" edgedefault="directed">\n')
      for i in tqdm.tqdm(range(num_nodes),desc='Nodes...'):
        f.write(f'    <node id="n{i}">\n')
        if node_features is not None:
          feat_str = ",".join(map(str, node_features[i]))
          f.write(f'      <data key="v_feat">{feat_str}</data>\n')
        f.write('    </node>\n')
      sources = edge_index[0]
      targets = edge_index[1]
      for idx in tqdm.tqdm(range(len(sources)),desc='Edges...'):
        # Write edge with optional weight
        if edge_weights is not None:
          w = edge_weights[idx].item() if hasattr(edge_weights[idx], "item") else edge_weights[idx]
          f.write(f'    <edge source="n{sources[idx]}" target="n{targets[idx]}">\n')
          f.write(f'      <data key="e_weight">{w}</data>\n')
          f.write('    </edge>\n')
        else:
          f.write(f'    <edge source="n{sources[idx]}" target="n{targets[idx]}"/>\n')
      f.write('  </graph>\n')
      f.write('</graphml>\n')                       

