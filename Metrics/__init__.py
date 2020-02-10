import torch


def avgNLL(loglikelihood, theta, X, y):
    L = loglikelihood(theta, X, y)
    M = torch.tensor(theta.shape[0]).type_as(theta)
    log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(M)
    return torch.mean(-log_posterior_predictive)