from src.ftm import rectangular_drum
from src.ftm import constants as FTM_constants
import functools
import torch

def Ploss_distance(theta_c,theta_r):
    """
    Return sqrt(||theta_r - theta_c||) = x*x.'

    theta_r,theta_c: reference and candidate
    """
    dt = torch.sub(theta_r,theta_c)
    return torch.matmul(dt,torch.transpose(dt,0,-1))

def Bruteforce_distance(theta_c,S_r,logscale,phi):
    """
    Return ||(phi o g)(theta_r) - (phi o g)(theta_c)|| = ||S_r - S_c||

    theta_r: reference
    S_r: (phi o g)(theta_r) precalculated
    """
    S_c = phi(rectangular_drum(theta_c, logscale=logscale,**FTM_constants))
    return torch.linalg.vector_norm(torch.sub(S_r, S_c))

def PerceptualKNN_distance(theta_c,theta_r,M_r):
    """
    Return (theta_r - theta_c).T * M(theta_r) * (theta_r - theta_c)

    theta_r,theta_c: reference and candidate
    M_r: M(theta_r) precalculated 
    """
    dt = torch.sub(theta_r,theta_c)
    return torch.matmul(torch.matmul(torch.transpose(M_r,0,1),dt),dt)

def distance_factory(distance_method,phi=None,logscale=None):
    """
    Return distance = f(theta_c,ref) with ref depending on the method used

    phi: perceptual distance function
    logscale: theta scale (True/False)
    distance_method: method for computing the distance (P-loss/Bruteforce/Perceptual-KNN)
    """
    if(distance_method=='P-loss'):
        return Ploss_distance
    elif(distance_method=='Bruteforce'):
        return functools.partial(Bruteforce_distance,logscale=logscale,phi=phi)
    elif(distance_method=='Perceptual-KNN'):
        return PerceptualKNN_distance