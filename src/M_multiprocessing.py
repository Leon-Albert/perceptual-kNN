import torch

_worker_M_func = None

def init_worker_M(M_factory_func, logscale, phi, device_str):
    """
    Initializes the M function once per worker process.
    """
    global _worker_M_func
    device = torch.device(device_str)
    _worker_M_func = M_factory_func(logscale, phi)

def compute_task_M(args):
    """
    Run M on a single row.
    """
    idx, data_row, device_str = args
    device = torch.device(device_str)
    
    with torch.no_grad():
        input_tensor = data_row.to(device)
        output = _worker_M_func(input_tensor)
        result = output.cpu()
        
    return idx, result