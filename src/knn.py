import torch
import functools
from src.distances import distance_factory
from src.jacobian import M_factory
from src.ftm import rectangular_drum
from src.ftm import constants as FTM_constants

def KnnG(DF,k,phi,logscale,distance_method,update_pb=None):
    """
    Return T_knn = [[theta_r1_1nn,theta_r1_2nn,...],[theta_r2_1nn,theta_r2_2nn,...],...]
    
    DF: dataframe of the points
    k: neighbours count
    distance_method: method for computing the distance (P-loss/Bruteforce/Perceptual-KNN)
    update_pb: fct for updating a progress bar outside of this scope (optional)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = DF.size(dim=0)

    T_knn = torch.zeros(n,k,DF.size(dim=1)).to(device)
    # Loop over each point as a reference
    for i_r in range(n):

        theta_r = DF[i_r,:]

        # Precompute the distance function for this reference & method
        if(distance_method=='Bruteforce'):
            S_r = phi(rectangular_drum(theta_r, logscale, **FTM_constants))
            distance = distance_factory(distance_method,phi=phi,logscale=logscale)

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
            T_dist = torch.zeros(DF.size(dim=0),1)
            for i_c in range(DF.size(dim=0)):
                theta_c = DF[i_c,:]
                T_dist[i_c] = distance(theta_c,S_r)
            T_dist = torch.transpose(T_dist,0,1).squeeze(0)

        if(update_pb):
            update_pb(n)    
        
        # Sort the distances to get the k nearest
        T_dist_sorted,i_c_sorted = torch.sort(T_dist)
        for i_k in range(k):
            T_knn[i_r,i_k,:] = DF[i_c_sorted[i_k],:]
                    
    return T_knn
