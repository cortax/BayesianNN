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

def NormalLogLikelihood(y_pred, y_data, sigma_noise):
    """
    Evaluation of a Normal distribution
    
    Parameters:
    y_pred (Tensor): tensor of size MxD
    y_data (Tensor): tensor of size 1xD
    sigma_noise (Scalar): standard deviation for the diagonal cov matrix

    Returns:
    logproba (Tensor): Mx1 vector of log probabilities
    """
    assert y_pred.shape[1] == y_data.shape[1]
    assert y_data.shape[0] == 1
    log_proba = log_norm(y_pred, y_data, sigma_noise).sum(dim=[1,2])
    assert log_proba.ndim == 1
    return log_proba

def logmvn01pdf(theta):
    """
    Evaluation of log proba with density N(0,I_n)

    Parameters:
    x (Tensor): Data tensor of size NxD

    Returns:
    logproba (Tensor): N-dimensional vector of log probabilities
    """
    dim = theta.shape[1]
    S = torch.ones(dim).type_as(theta)
    mu = torch.zeros(dim).type_as(theta)
    n_x = theta.shape[0]

    H = S.view(dim, 1, 1).inverse().view(1, 1, dim)
    d = ((theta-mu.view(1, dim))**2).view(n_x, dim)
    const = 0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
    return -0.5*(H*d).sum(2).squeeze()-const


