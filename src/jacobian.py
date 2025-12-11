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
    return M_from_G(G(theta).unsqueeze(0))

def M_factory(logscale,Phi):
    """
    Return M = f(theta)
    
    phi: perceptual distance function
    logscale: theta scale (True/False)
    FTM_constants: constants for the ftm synth
    """
    S_from_theta = S_factory(logscale,Phi)
    G = torch.func.jacfwd(S_from_theta)
    #G = functools.partial(torch.autograd.functional.jacobian, func=S_from_theta, create_graph=False) #,strategy="forward-mode",vectorize=True) 
    return functools.partial(M_forward,G=G)


