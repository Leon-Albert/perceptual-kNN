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

  
  '''General Private Methods'''

  def scalar_product_pnp(self, theta1, theta2, thetaA, M):
    v1 = theta1 - thetaA
    v2 = theta2 - thetaA
    return torch.einsum('...i,...ij,...j->...', v1, M, v2)

  def _distance_pnp(self,theta1,theta2,M):
    #'...' in einsum to handle both batched and single inputs
    return self.scalar_product_pnp(theta1,theta1,theta2,M)

  def _M_from_index(self,Li):
    LM = []
    for i in Li:
      M = self.M_func(self.DF[i,:])
      LM.append(M)
    return torch.stack(LM,dim=0).to(self.device)
  
  def _V_from_A(self):
    V = [[] for l in range(torch.max(self.H)+1)]
    for i in range(self.H[self.A].size(dim=0)):
      V[self.H[self.A][i].item()].append(i)
    return V
  
  def _allocation(self):
    hubs_idx = self.H.flatten()
    K = hubs_idx.size(0)

    diff = self.DF.unsqueeze(1) - self.DF[hubs_idx].unsqueeze(0)
    distances = torch.einsum('nki,kij,nkj->nk', diff, self.M_hub, diff)
        
    Ai_index = torch.argmin(distances, dim=1)
    
    # Force hubs to be allocated to themselves
    hub_range = torch.arange(K, device=self.device)
    Ai_index.scatter_(0, hubs_idx, hub_range)
    
    return Ai_index
  

  '''Hub Selection Private Methods'''
  
  def _criteria(self, pair, k, max_test=200):
    l1 = torch.tensor(list(self.V[pair[0]]), device=self.device).long()
    l2 = torch.tensor(list(self.V[pair[1]]), device=self.device).long()
    
    idx1 = torch.randperm(len(l1))[:max_test]
    idx2 = torch.randperm(len(l2))[:max_test]
    n_indices = l1[idx1] 
    m_indices = l2[idx2] 

    N_grid, M_grid = torch.meshgrid(n_indices, m_indices, indexing='ij')
    N_flat = N_grid.reshape(-1)
    M_flat = M_grid.reshape(-1)
    K_val = torch.full_like(N_flat, k)

    df_n = self.DF[N_flat]
    df_m = self.DF[M_flat]
    df_k = self.DF[K_val]

    m_hub_k = self.M_hub[self.A[K_val]]
    m_hub_n = self.M_hub[self.A[N_flat]]
    m_hub_m = self.M_hub[self.A[M_flat]]

    d0 = self._distance_pnp(df_n,df_k,m_hub_k)
    d1 = self._distance_pnp(df_k,df_m,m_hub_k)
    d2 = self._distance_pnp(df_n,df_m,m_hub_n)
    d3 = self._distance_pnp(df_m,df_n,m_hub_m)

    Rk_total = torch.sum((d0 + d1) / (d2 + d3))
    return Rk_total.item()

  def _choose_hub(self):
    Lkmax_pair = []
    LRkmax_pair = []
    for i_pair in range(len(self.LP)):
      pair = self.LP[i_pair]
      candidats = torch.cat((torch.tensor(self.V[pair[0]]),torch.tensor(self.V[pair[1]])),0)
      LRk = []
      for k in candidats:
        LRk.append(self._criteria(pair,k))
      Lkmax_pair.append(torch.argmax(torch.tensor(LRk),dim=0))
      LRkmax_pair.append(LRk[Lkmax_pair[-1]])
    return Lkmax_pair[torch.argmax(torch.tensor(LRkmax_pair),dim=0)]
  
  def _new_hub(self):
    k = self._choose_hub()
    if isinstance(k, torch.Tensor):
        k = k.item()
    
    new_h_idx = torch.tensor([k], device=self.device, dtype=torch.long)
    self.H = torch.cat((self.H, new_h_idx), dim=0)
    
    new_M = self._M_from_index([k])
    self.M_hub = torch.cat((self.M_hub, new_M), dim=0)
    
    self.A = self._allocation()
    
    self.V = self._V_from_A()

  def _iter(self):
    old_A = self.A.clone()
    self._new_hub()
    new_LP = []
    changed_mask = (old_A != self.A)
    if changed_mask.any():
        changed_from = self.H[old_A][changed_mask]
        changed_to = self.H[self.A][changed_mask]
        pairs = torch.stack([changed_from, changed_to], dim=1)
        unique_pairs = torch.unique(pairs, dim=0)
        new_LP = [tuple(p.tolist()) for p in unique_pairs]
    else:
      pass #Should never happen in theory
    return new_LP
  
  
  '''Public Methods'''
  
  def initialize(self):
    perm = torch.randperm(self.N, device=self.device)
    self.H = perm[:2] 
    
    self.M_hub = self._M_from_index(self.H.tolist())
    
    self.A = self._allocation()
    self.V = self._V_from_A()
    
    self.LP = [(self.H[0].item(), self.H[1].item())]
  
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
    self.Histo = []
    for i in tqdm.tqdm(range(self.max__iter_hubs),desc="Constructing hubs..."):

      self.Histo.append(torch.bincount(self.A))

      self._iter()

  def construct_edges(self, k, batch_size=128):
    M_all = self.M_hub[self.A].to(self.device)  
    sources_list = []
    targets_list = []
    weights_list = []

    def compute_row(theta_c,M_c,theta_r,M_r):
      # M_r (single matrix) and M_c (batched matrix)
      d1 = self._distance_pnp(theta_c, theta_r, M_r)
      d2 = self._distance_pnp(theta_r, theta_c, M_c)
      return (d1 + d2) / 2

    for i in tqdm.tqdm(range(0, self.N, batch_size), desc='Constructing edges...'):
      start = i
      end = min(i + batch_size, self.N)          
      batch_theta = self.DF[start:end]   
      batch_M = M_all[start:end]     
      dists_batch = []

      for b in range(end - start):
        d_row = compute_row(self.DF, M_all, batch_theta[b], batch_M[b]).cpu()
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

