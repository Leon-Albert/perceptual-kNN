import torch
import tqdm
import torch_geometric as tg
import pyarrow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq



class graph_constructor: 

  def __init__(self,DF, max__iter_hubs, M_func):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.DF = DF.to(self.device)
    self.N = DF.size(dim=0)
    self.max__iter_hubs = max__iter_hubs
    self.M_func = M_func

  
  '''General Private Methods'''

  def _distance_pnp(self,theta1,theta2,M):
    #'...' in einsum to handle both batched and single inputs
    v = theta1 - theta2
    return torch.einsum('...i,...ij,...j->...', v, M, v)

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
 
    min_dists, A = torch.min(distances, dim=1)
    
    self.dists_to_hub = min_dists
    
    # hubs have to be allocated to themselves
    hub_range = torch.arange(K, device=self.device)
    A.scatter_(0, hubs_idx, hub_range)
    
    return A
  

  '''Hub Selection Private Methods'''
  
  def _choose_hub(self):

    counts = torch.bincount(self.A)
    dominant_hub_idx = torch.argmax(counts).item()
    
    # Get indices and their distances
    mask = (self.A == dominant_hub_idx)
    candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
    candidate_dists = self.dists_to_hub[mask]
  
    sorted_vals, sorted_idx = torch.sort(candidate_dists, descending=True)
    
    n_candidates = len(candidate_indices)
    
    # window of valid candidates, we ignore the top 10% (outliers) and bottom 30% (too close to center)
    start = int(n_candidates * 0.10) 
    end = int(n_candidates * 0.30)
    
    # Safety check for small clusters
    if start >= end:
        picked_idx_local = 0
    else:
        picked_idx_local = torch.randint(start, end, (1,)).item()
        
    original_idx_in_candidates = sorted_idx[picked_idx_local]
    new_hub_id = candidate_indices[original_idx_in_candidates].item()
    
    return new_hub_id

  def _iter(self):
    k = self._choose_hub()
    new_h_idx = torch.tensor([k], device=self.device, dtype=torch.long)
    self.H = torch.cat((self.H, new_h_idx), dim=0)

    new_M = self._M_from_index([k])
    self.M_hub = torch.cat((self.M_hub, new_M), dim=0)

    self.A = self._allocation()
    self.V = self._V_from_A()

  
  '''Public Methods'''
  
  def initialize(self,nbr_of_initial_hubs=1):
    perm = torch.randperm(self.N, device=self.device)
    self.H = perm[:nbr_of_initial_hubs] 
    
    self.M_hub = self._M_from_index(self.H.tolist())
    
    self.A = self._allocation()
    self.V = self._V_from_A()
      
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
    i=0
    while i <= self.max__iter_hubs:
      print(f"Constructing - i={i}")
      self._iter()
      i+=1

      if i%100 == 0:
        counts = torch.bincount(self.A, minlength=self.H.size(0))
        
        # Identify hubs that have more than 1 point (themselves)
        keep_mask = counts > 1
        
        # If all hubs are lonely, or only 20 or less are lonely we skip (so we're not stuck at the end)
        if not keep_mask.any() or self.max__iter_hubs - len(keep_mask)<20:
            continue
            
        # Update Hub indices and their corresponding M
        self.H = self.H[keep_mask]
        self.M_hub = self.M_hub[keep_mask]
        
        # Re-allocate all points to the remaining hubs
        self.A = self._allocation()
        self.V = self._V_from_A()

        i=len(keep_mask)


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

    # coalesce removes duplicates (i in j's knn but also j in i's knn)
    self.edge_index, self.edge_attr = tg.utils.coalesce(edge_index, all_weights, reduce="mean")

  
  def construct_edges_param(self, k):
    dists = torch.cdist(self.DF, self.DF, p=2)
    vals, indices = torch.topk(dists, k=k+1, dim=1, largest=False)
    
    sources = torch.arange(self.N, device=self.DF.device).unsqueeze(1).expand(-1, k).reshape(-1)
    targets = indices[:, 1:].reshape(-1)
    weights = vals[:, 1:].reshape(-1)

    edge_index = torch.stack([
        torch.cat([sources, targets]),
        torch.cat([targets, sources])
    ], dim=0)
    
    edge_attr = torch.cat([weights, weights])

    # coalesce removes duplicates (i in j's knn but also j in i's knn)
    self.edge_index, self.edge_attr = tg.utils.coalesce(edge_index, edge_attr, reduce="mean")


  def construct_edges_groundtruth(self, k, path_S, query_batch_size=256, ref_block_size=512):
    parquet_file = pq.ParquetFile(path_S)
    N = parquet_file.metadata.num_rows
    D = parquet_file.metadata.num_columns
    sources_list = []
    targets_list = []
    weights_list = []
    global_query_idx = 0
    query_pbar = tqdm.tqdm(total=N, desc='Total Progress')
    
    for batch_arrow in parquet_file.iter_batches(batch_size=query_batch_size):
        batch_q_np = batch_arrow.to_pandas().values
        batch_q = torch.tensor(batch_q_np, dtype=torch.float32).to(self.device)
        curr_q_size = batch_q.size(0)

        batch_distances = torch.zeros((curr_q_size, N), device=self.device)

        global_ref_idx = 0
        for ref_arrow in parquet_file.iter_batches(batch_size=ref_block_size):
            batch_r_np = ref_arrow.to_pandas().values
            batch_r = torch.tensor(batch_r_np, dtype=torch.float32).to(self.device)
            curr_ref_size = batch_r.size(0)

            dists = torch.cdist(batch_q, batch_r, p=2)
            batch_distances[:, global_ref_idx : global_ref_idx + curr_ref_size] = dists
            
            global_ref_idx += curr_ref_size
            del batch_r

        vals, indices = torch.topk(batch_distances, k=k+1, dim=1, largest=False)
        
        vals = vals.cpu()
        indices = indices.cpu()

        for i in range(curr_q_size):
            u_id = global_query_idx + i
            
            v_indices = indices[i]
            v_weights = vals[i]

            mask = v_indices != u_id
            final_v = v_indices[mask][:k]
            final_w = v_weights[mask][:k]

            u_tensor = torch.full((final_v.size(0),), u_id, dtype=torch.long)

            sources_list.append(u_tensor)
            targets_list.append(final_v)
            weights_list.append(final_w)

            sources_list.append(final_v)
            targets_list.append(u_tensor)
            weights_list.append(final_w)

        global_query_idx += curr_q_size
        query_pbar.update(curr_q_size)
        
        del batch_q, batch_distances
    query_pbar.close()

    edge_index = torch.stack([
        torch.cat(sources_list), 
        torch.cat(targets_list)
    ], dim=0)
    edge_attr = torch.cat(weights_list)

    # coalesce removes duplicates (i in j's knn but also j in i's knn)
    self.edge_index, self.edge_attr = tg.utils.coalesce(edge_index, edge_attr, reduce="mean")


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


  def save_hubs(self,folder_path):
    torch.save(self.H,folder_path+'H.pt')
    torch.save(self.M_hub,folder_path+'M_hub.pt')
    torch.save(self.A,folder_path+'A.pt')
    torch.save(self.V,folder_path+'V.pt')

  def load_hubs(self,folder_path):
    self.H = torch.load(folder_path+'H.pt')
    self.M_hub = torch.load(folder_path+'M_hub.pt')
    self.A = torch.load(folder_path+'A.pt')
    self.V = torch.load(folder_path+'V.pt')


  '''Methods for quality checking the hubs'''

  def analyze_hub_quality(self):
    num_hubs = self.H.size(0)
    counts = torch.bincount(self.A, minlength=num_hubs)
    
    hub_stats = []
    for i in range(num_hubs):
      mask = (self.A == i)
      pop_size = counts[i].item()
      
      # Default values for empty or single-point hubs
      avg_dist, max_dist, std_dist = 0.0, 0.0, 0.0
      
      if pop_size > 0:
          assigned_dists = self.dists_to_hub[mask]
          avg_dist = assigned_dists.mean().item()
          max_dist = assigned_dists.max().item()
          
          # Standard deviation only exists if there is more than 1 point
          if pop_size > 1:
              std_dist = assigned_dists.std().item()
          
      hub_stats.append({
          'index': i,
          'size': pop_size,
          'avg_dist': avg_dist,
          'max_dist': max_dist,
          'std_dist': std_dist
      })
  
    return hub_stats

  def print_quality_report(self):
    """Prints the distribution of hub sizes and their respective approximation quality."""
    stats = self.analyze_hub_quality()
    
    print(f"\n{'Hub Index':<10} | {'Size':<8} | {'Avg Dist':<12} | {'Std Dev':<12} | {'Max Dist':<12}")
    print("-" * 70)
    
    # Sort by size descending
    sorted_stats = sorted(stats, key=lambda x: x['size'], reverse=True)
    
    for s in sorted_stats:
        print(f"{s['index']:<10} | {s['size']:<8} | {s['avg_dist']:<12.4e} | {s['std_dist']:<12.4e} | {s['max_dist']:<12.4e}")


