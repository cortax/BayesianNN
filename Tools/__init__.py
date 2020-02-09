import torch
import math


def log_norm(x, mu, std):
    """
    Evaluation of 1D normal distribution on tensors

    Parameters:
        x (Tensor): Data tensor of size B X S 
        mu (Tensor): Mean tensor of size S 
        std (Float): Positive scalar (standard deviation)

    Returns:
        logproba (Tensor): Same size as x with logproba(b,i)=log p(x(b,i),mu(i),std)
    """
    B = x.shape[0]
    S = x.shape[1]
    var = torch.as_tensor(std**2).type_as(x)
    d = (x.view(B, S, 1)-mu.view(1, S, 1))**2
    c = 2*math.pi*var
    return -0.5 * (1/(var))*d - 0.5 * c.log()
