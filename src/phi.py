import auraloss
import functools
import torch

def STFT_magnitude(x,fft_size,hop_size,win_length,window):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window(win_length).to(device),
        return_complex=True,
        center=False
    )
    return torch.abs(x_stft)

def STFT_loss(input,target,fft_size,hop_size,win_length,window):
    # Compute the STFTs
    x_mag = STFT_magnitude(input ,fft_size,hop_size,win_length,window)
    y_mag = STFT_magnitude(target,fft_size,hop_size,win_length,window)
    # Compute the L1 norms
    eps = 1e-8
    log_mag_loss = torch.sum(torch.abs(torch.log(x_mag+eps)-torch.log(y_mag+eps)))
    lin_mag_loss = torch.sum(torch.abs(x_mag-y_mag))
    return (log_mag_loss + lin_mag_loss)

def STFT_losses_mean(x, y, stft_losses):
    # Compute the mean of the losses for each hyperparameters set
    mrstft_loss = 0.0
    for f in stft_losses:
        mrstft_loss += f(x.unsqueeze(0).to(torch.float), y.unsqueeze(0).to(torch.float))
    mrstft_loss /= len(stft_losses)
    return mrstft_loss

def MSS_factory(fft_sizes = [1024, 2048, 512],hop_sizes = [120, 240, 50],win_lengths = [600, 1200, 240]):
    stft_losses = []
    for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
        stft_losses.append(functools.partial(STFT_loss,fft_size=fs,hop_size=ss,win_length=wl,window=torch.hann_window))
    return functools.partial(STFT_losses_mean,stft_losses=stft_losses)

def MSS_ref_factory(x_r):
    """
    Return MSS = f(x_c)
    
    theta_r: reference
    """    
    MSS = MSS_factory()
    return functools.partial(MSS,y=x_r)