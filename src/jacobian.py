import functools
import src.ftm as ftm
from src.ftm import constants as FTM_constants
import torch

def x_from_theta(theta, logscale):
    """
    Return x = g(theta)
    """
    return ftm.rectangular_drum(theta, logscale, **FTM_constants)
    
def S_forward(theta, logscale, Phi):
    """
    Return S = (phi o g)(theta)
    """
    g = functools.partial(x_from_theta,logscale=logscale)
    x = g(theta)
    return Phi(x)
    
def S_factory(logscale, Phi):
    """
    Return S = (phi o g)
    """ 
    return functools.partial(S_forward, logscale=logscale, Phi=Phi)      

def M_from_G(G):
    return torch.matmul(torch.transpose(G,0,1),G)

def M_forward(theta, G):
    """
    Return M(theta,G) = G(theta).T * G(theta) = grad(Phi o g)(theta0).T * grad(Phi o g)(theta0)
    """
    return M_from_G(G(theta))

def M_factory(logscale,Phi):
    """
    Return M = f(theta)

    We use a forward strategy for the gradient calculation based on the dimensions at the different steps of the pipeline 
    2 different functions are available in Torch, one might work better depending on Phi and the hardware, try both to be sure
    
    phi: perceptual distance function
    logscale: theta scale (True/False)
    """
    S_from_theta = S_factory(logscale,Phi)
    G = torch.func.jacfwd(S_from_theta)
    #G = functools.partial(torch.autograd.functional.jacobian, func=S_from_theta, create_graph=False,strategy="forward-mode",vectorize=True) 
    return functools.partial(M_forward,G=G)

def compute_all_S(DF,Phi,logscale):
    """
    Return T_S with T_S[i,:] = S(DF[i,:]) = (phi o g)(DF[i,:])
    
    DF: tensor with the dataset (DF[i,:] = theta_i)
    phi: perceptual distance function
    logscale: theta scale (True/False)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = DF.size(dim=0)

    S = Phi(ftm.rectangular_drum(DF[0,:], logscale, **FTM_constants))
    T_S = torch.zeros(n,S.size(dim=0)).to(device)
    
    T_S[0,:] = S
    print("S computation: ",1,"/",n)

    #TODO improve this somehow ?
    for id in range(1,n):
        T_S[id,:] = Phi(ftm.rectangular_drum(DF[id,:], logscale, **FTM_constants))
        print("S computation: ",id+1,"/",n)

    return T_S