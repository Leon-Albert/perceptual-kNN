import torch

def Ploss_distance(theta_c,theta_r):
    """
    Return sqrt(||theta_r - theta_c||) = x*x.'

    theta_r,theta_c: reference and candidate points
    """
    dt = torch.sub(theta_r,theta_c)
    return torch.dot(dt,dt)

def Bruteforce_distance(S_c,S_r):
    """
    Return ||(phi o g)(theta_r) - (phi o g)(theta_c)|| = ||S_r - S_c|| = x*x.'

    S_c: (phi o g)(theta_c) precalculated
    S_r: (phi o g)(theta_r) precalculated
    """
    dt = torch.sub(S_r, S_c)
    return torch.dot(dt,dt)

def PNP_distance(theta_c,theta_r,M_r):
    """
    Return (theta_r - theta_c).T * M(theta_r) * (theta_r - theta_c)

    theta_r,theta_c: reference and candidate points
    M_r: M(theta_r) precalculated 
    """
    dt = torch.sub(theta_r,theta_c)
    return torch.matmul(torch.matmul(torch.transpose(M_r,0,1),dt),dt)

def distance_factory(distance_method):
    """
    Return distance = f(candidate,reference) with types depending on the method used

    distance_method: method for computing the distance (P-loss/Bruteforce/PNP)
    """
    if(distance_method=='P-loss'):
        return Ploss_distance
    elif(distance_method=='Bruteforce'):
        return Bruteforce_distance
    elif(distance_method=='PNP'):
        return PNP_distance