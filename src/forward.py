import ftm
import functools

def x_from_theta(theta, logscale):
    """Compute x = g(theta)"""
    x = ftm.rectangular_drum(theta, logscale, **ftm.constants)
    return x

def S_from_x(x, Phi_operator, Phi_params):
    """Compute S = Phi(x)"""
    Phi = Phi_operator(Phi_params)
    S = Phi(x)
    return S
        
def pnp_forward(theta, logscale, Phi_operator, Phi_params):
    """Compute S = (phi o g)(theta)"""
    g = functools.partial(x_from_theta,logscale=logscale)
    Phi = functools.partial(S_from_x, Phi_operator=Phi_operator, Phi_params=Phi_params)
    S =  Phi(g(theta))        
    return S

def pnp_forward_factory(logscale, Phi_operator, Phi_params):
    """Compute S = (phi o g)""" 
    return functools.partial(pnp_forward, logscale=logscale, Phi_operator=Phi_operator, Phi_params=Phi_params)        


