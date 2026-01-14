import torch


def get_device_from_str(device_str):
    """ Helper function for getting device from string. """
    if device_str not in ['auto', 'cpu', 'cuda']:
        raise ValueError('Device string must be one of "auto", "cpu" or "cuda"')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if device_str == 'auto' else device_str
