import functools
import src.ftm as ftm
import torch

def x_from_theta(theta, logscale, FTM_constants):
    """
    Return x = g(theta)
    """
    return ftm.rectangular_drum(theta, logscale, **FTM_constants)
    
def S_forward(theta, logscale, Phi, FTM_constants):
    """
    Return S = (phi o g)(theta)
    """
    g = functools.partial(x_from_theta,logscale=logscale,FTM_constants=FTM_constants)
    x = g(theta)
    return Phi(x)
    
def S_factory(logscale, Phi, FTM_constants):
    """
    Return S = (phi o g)
    """ 
    return functools.partial(S_forward, logscale=logscale, Phi=Phi, FTM_constants=FTM_constants)      

def M_from_G(G):
    return torch.matmul(torch.transpose(G,0,1),G)

def M_forward(theta, G):
    """
    Return M(theta,G) = G(theta).T * G(theta) = grad(Phi o g)(theta0).T * grad(Phi o g)(theta0)
    """
    return M_from_G(G(theta).unsqueeze(0))

def M_factory(logscale,Phi,FTM_constants):
    """
    Return M = f(theta)
    
    phi: perceptual distance function
    logscale: theta scale (True/False)
    FTM_constants: constants for the ftm synth
    """
    S_from_theta = S_factory(logscale,Phi,FTM_constants)
    G = torch.func.jacfwd(S_from_theta)
    return functools.partial(M_forward,G=G)


