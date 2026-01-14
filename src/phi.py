import torch
from kymatio.torch import TimeFrequencyScattering1D

jtfs_params = dict(
            J = 14, #scale
            shape = (2**16, ), 
            Q = 12, #filters per octave, frequency resolution
            T = 2**13,  #local averaging
            F = 2,
            max_pad_factor=1,
            max_pad_factor_fr=1,
            average = True,
            average_fr = True,
)

def JTFS_forward(input):
    """
    Return phi(x)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jtfs_operator = TimeFrequencyScattering1D(**jtfs_params, out_type="list").to(device)
    Sx_list = jtfs_operator.scattering(input)

    Sx_array = torch.cat([path['coef'].flatten() for path in Sx_list])
    # apply "stable" log transformation
    # the number 1e3 is ad hoc and of the order of 1/mu where mu=1e-3 is the
    # median value of Sx across all paths
    log1p_Sx = torch.log1p(Sx_array*1e3)

    return log1p_Sx