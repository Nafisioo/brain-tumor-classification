import torch


def get_device():
    """
    Select best available device.

    Priority:
    1. CUDA
    2. Apple Silicon MPS
    3. CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
