import time
from src.ftm import rectangular_drum
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def find_neighbour(DataFrame_path,ref_index_list,k,dist,return_time=False,show_progress=False):
    # Initialize 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.from_numpy(pd.read_csv(DataFrame_path, index_col=0).to_numpy()).to(device).to(float)
    datasize = data.size(dim=0)
    dist_i = torch.zeros(2,len(ref_index_list)*k).to(device)
    dist_v = torch.zeros(1,len(ref_index_list)*k).to(device)[0]

    # Build tensor of the reference points (for the candidates it's data)
    tensor_refs = torch.zeros(len(ref_index_list),data.size(dim=1)).to(device).to(float)
    for i in range(len(ref_index_list)):
        id = ref_index_list[i]
        tensor_refs[i,:] = data[id,:]

    # Find the distances for each t_ref to all the candidates. 
    if show_progress:
        iterate = tqdm(range(tensor_refs.size(dim=0)))
    else:
        iterate = range(tensor_refs.size(dim=0))

    for i_ref in iterate:
        
        dist_tensor = dist(data,tensor_refs[i_ref,:])
        
        # Add the k shortest distances to the matrix
        dist_tensor,sort_indices = torch.sort(dist_tensor)

        dist_i[0,k*i_ref:k*(i_ref+1)] = i_ref*torch.ones(1,k)
        dist_i[1,k*i_ref:k*(i_ref+1)] = sort_indices[0:k].to(device)
        dist_v[k*i_ref:k*(i_ref+1)] = dist_tensor[0:k]
        
        # Remove the ref point from the futur candidates 
        data = torch.cat((data[:i_ref],data[i_ref+1:]))

    return dist_i.to(int),dist_v