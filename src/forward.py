import functools
import ftm as ftm

def x_from_theta(theta, logscale):
    """Compute x = g(theta)"""
    x = ftm.rectangular_drum(theta, logscale, **ftm.constants)
    return x
        
def pnp_forward(theta, logscale, Phi):
    """Compute S = (phi o g)(theta)"""
    g = functools.partial(x_from_theta,logscale=logscale)
    x = g(theta)
    S = Phi(x)
    return S

def pnp_forward_factory(logscale, Phi):
    """Compute S = (phi o g)""" 
    return functools.partial(pnp_forward, logscale=logscale, Phi=Phi)        


