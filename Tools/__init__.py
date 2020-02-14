import torch
import math

#TODO vérifier si ces fonctions ne doivent pas prendre device en argument
# device ajouter pour logmvn01pdf


def log_norm(x, mu, std):
    """
    Evaluation of 1D normal distribution on tensors

    Parameters:
        x (Tensor): Data tensor of size B X S X 1
        mu (Tensor): Mean tensor of size S X 1
        std (Float): Positive scalar (standard deviation)

    Returns:
        logproba (Tensor): size B X S X 1 with logproba(b,i)=[log p(x(b,i),mu(i),std)]
    """
    assert x.shape[1] == mu.shape[0]
    assert x.shape[2] == mu.shape[1]
    assert mu.shape[1] == 1
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
    y_pred (Tensor): tensor of size M X N X 1
    y_data (Tensor): tensor of size N X 1
    sigma_noise (Scalar): std for point likelihood: p(y_data | y_pred, sigma_noise) Gaussian N(y_pred,sigma_noise)

    Returns:
    logproba (Tensor):  (raw) size M X N , with logproba[m,n]= p(y_data[n] | y_pred[m,n], sigma_noise)                        (non raw) size M , logproba[m]=sum_n logproba[m,n]
    """
# assert taken care of by log_norm
#    assert y_pred.shape[1] == y_data.shape[0]
#    assert y_pred.shape[2] == y_data.shape[1]
#    assert y_data.shape[1] == 1
    log_proba = log_norm(y_pred, y_data, sigma_noise)
    return log_proba.squeeze(-1)

def logmvn01pdf(theta, device):
    """
    Evaluation of log proba with density N(0,I_n)

    Parameters:
    x (Tensor): Data tensor of size NxD

    Returns:
    logproba (Tensor): size N, vector of log probabilities
    """
    dim = theta.shape[1]
    S = torch.ones(dim).type_as(theta).to(device)
    mu = torch.zeros(dim).type_as(theta).to(device)
    n_x = theta.shape[0]

    H = S.view(dim, 1, 1).inverse().view(1, 1, dim)
    d = ((theta-mu.view(1, dim))**2).view(n_x, dim)
    const = 0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
    return -0.5*(H*d).sum(2).squeeze()-const


