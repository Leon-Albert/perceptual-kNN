import torch
import functools
from src.distances import distance_factory
from src.jacobian import M_factory
from src.jacobian import compute_all_S

def Knn(DF,i_r,k,phi,logscale,distance_method):
    """
    Return T_knn = [[theta_r1_1nn,theta_r1_2nn,...],[theta_r2_1nn,theta_r2_2nn,...],...]
    
    DF: dataframe of the points*
    i_r: if of the reference point
    k: neighbours count
    distance_method: method for computing the distance (P-loss/Bruteforce/Perceptual-KNN)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    T_knn = torch.zeros(k,DF.size(dim=1)).to(device)

    theta_r = DF[i_r,:]

    # Compute everything needed for this method
    if(distance_method=='Bruteforce'):
        T_S = compute_all_S(DF,phi,logscale) #TODO Should be pre-computed once for the full dataset instead
        distance = distance_factory(distance_method)

    elif(distance_method=='Perceptual-KNN'):
        M = M_factory(logscale,phi)
        M_r = M(theta_r)
        distance = distance_factory(distance_method)

    elif(distance_method=='P-loss'):
        distance = distance_factory(distance_method)

    # Compute distances to each candidate
    if(distance_method=='P-loss'):
        distance_batch = torch.func.vmap(functools.partial(distance,theta_r=theta_r))
        T_dist = distance_batch(DF)

    elif(distance_method=='Perceptual-KNN'):
        distance_batch = torch.func.vmap(functools.partial(distance,theta_r=theta_r,M_r=M_r))
        T_dist = distance_batch(DF)

    elif(distance_method=='Bruteforce'):
        distance_batch = torch.func.vmap(functools.partial(distance,S_r=T_S[i_r,:]))
        T_dist = distance_batch(T_S)
    
    # Sort the distances to get the k nearest
    T_dist_sorted,i_c_sorted = torch.sort(T_dist)
    for i_k in range(k):
        T_knn[i_k,:] = DF[i_c_sorted[i_k],:]
                    
    return T_knn
