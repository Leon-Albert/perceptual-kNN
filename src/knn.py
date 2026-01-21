import torch
import functools
from src.distances import distance_factory
from src.jacobian import M_factory
from src.dataset_utils import S_ds_read_given_rows
import pyarrow.parquet as pq
import numpy as np
import tqdm


def Knn(DF,i_r,k,phi,logscale,distance_method,S_data_path):
    """
    Return T_knn = [[theta_r1_1nn,theta_r1_2nn,...],[theta_r2_1nn,theta_r2_2nn,...],...]
    
    DF: dataframe of the points*
    i_r: if of the reference point
    k: neighbours count
    distance_method: method for computing the distance (P-loss/Bruteforce/PNP)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    theta_r = DF[i_r,:]

    # Compute distances to each candidate
    if(distance_method=='P-loss'):
        # Setup
        distance = distance_factory(distance_method)
        distance_batch = torch.func.vmap(functools.partial(distance,theta_r=theta_r))
        # Computing
        T_dist = distance_batch(DF)

    elif(distance_method=='PNP'):
        # Setup
        M = M_factory(logscale,phi)
        M_r = M(theta_r)
        distance = distance_factory(distance_method)
        distance_batch = torch.func.vmap(functools.partial(distance,theta_r=theta_r,M_r=M_r))
        # Computing
        T_dist = distance_batch(DF)

    elif(distance_method=='Bruteforce'):
        # Setup
        parquet_file = pq.ParquetFile(S_data_path)
        S_r_T = S_ds_read_given_rows(S_data_path, [i_r]) #This takes a lot of time but we do it once so..
        S_r = S_r_T[0,:].to(device)
        distance = distance_factory(distance_method)
        distance_batch = torch.func.vmap(functools.partial(distance,S_r=S_r))
        # Computing
        distances = []
        for i in tqdm.tqdm(range(parquet_file.num_row_groups), desc="Bruteforcing in batch"):
            table = parquet_file.read_row_group(i)
            S_batch_cpu = torch.from_numpy(np.array(table.drop(["row_id"])))
            # pin_memory to speed things up a bit, non_blocking=True allows the next read to start while this transfer happens
            S_batch = S_batch_cpu.pin_memory().to(device, non_blocking=True)
            dists = distance_batch(S_batch)
            distances.append(dists.flatten().cpu())

        T_dist = torch.cat(distances)


    # Sort the distances to get the k nearest
    T_dist_sorted,i_c_sorted = torch.sort(T_dist)
    T_knn = torch.zeros(k,DF.size(dim=1)).to(device)

    for i_k in range(k):
        T_knn[i_k,:] = DF[i_c_sorted[i_k],:]  
    return T_knn
