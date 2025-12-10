import torch
from src.distances import distance_factory
from src.jacobian import M_factory
from src.ftm import rectangular_drum

def KnnG(DF,k,phi_factory,logscale,FTM_constants,distance_method,update_pb=None):
    """
    Return T_knn = [[theta_r1_1nn,theta_r1_2nn,...],[theta_r2_1nn,theta_r2_2nn,...],...]
    
    DF: dataframe of the points
    k: neighbours count
    distance_method: method for computing the distance (P-loss/Bruteforce/Perceptual-KNN)
    update_pb: fct for updating a progress bar outside of this scope (optional)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute tensor of the neighbours TODO improve this with knn tricks
    T_knn = torch.zeros(DF.size(dim=0),k,DF.size(dim=1)).to(device).to(float)
    for i_r in range(DF.size(dim=0)):
        theta_r = DF[i_r,:] 
        x_r = rectangular_drum(theta_r, logscale, **FTM_constants)

        # Initialize needed functions that depends on theta_r
        phi = phi_factory(x_r)
        distance = distance_factory(phi,logscale,FTM_constants,distance_method)
        
        # Precompute the distance parameters if needed by method
        if(distance_method=='Bruteforce'):
            S_r = phi(x_r)
        elif(distance_method=='Perceptual-KNN'):
            M = M_factory(logscale,phi,FTM_constants)
            M_r = M(theta_r)

        # Compute distances to each candidate
        T_dist = torch.zeros(DF.size(dim=0),1)
        for i_c in range(DF.size(dim=0)):
            theta_c = DF[i_c,:]

            if(distance_method=='P-loss'):
                T_dist[i_c] = distance(theta_c,theta_r)
            elif(distance_method=='Bruteforce'):
                T_dist[i_c] = distance(theta_c,S_r)
            elif(distance_method=='Perceptual-KNN'):
                T_dist[i_c] = distance(theta_c,theta_r,M_r)

            if(update_pb):
                update_pb(1)

        # Sort the distances to get the k nearest
        T_dist_sorted,i_c_sorted = torch.sort(torch.transpose(T_dist,0,1))
        for i_k in range(k):
            T_knn[i_r,i_k,:] = DF[i_c_sorted[0,i_k],:]
                    
    return T_knn
