import torch


# def MSE(theta,model,x,y,inv_transform,device):
#     n_samples=x.shape[0]
#     y_pred =inv_transform(model(x, theta)).mean(0)
#     se=(y_pred-y.view(n_samples,1))**2
#     return torch.std_mean(se)


def avgNLL(loglikelihood, theta, X, y):
    L = loglikelihood(theta, X, y)
    M = torch.tensor(theta.shape[0]).type_as(theta)
    log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(M)
    return torch.mean(-log_posterior_predictive)