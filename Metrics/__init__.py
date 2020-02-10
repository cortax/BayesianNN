import torch


# def MSE(theta,model,x,y,inv_transform,device):
#     n_samples=x.shape[0]
#     y_pred =inv_transform(model(x, theta)).mean(0)
#     se=(y_pred-y.view(n_samples,1))**2
#     return torch.std_mean(se)

def MSE(theta,model,X,y,inv_transform,device):
    n_samples=x.shape[0]
    y_pred =inv_transform(model(x, theta)).mean(0)
    se=(y_pred-y.view(n_samples,1))**2
    return torch.std_mean(se)

#NLPD from Quinonero-Candela and al.
# average Negative Log Likelihood
# the average negative log predictive density (NLPD) of the true targets

def NLPD(theta,model, x, y, sigma_noise,inv_transform,device):
    y_pred =inv_transform(model(x, theta))
    L = _log_norm(y_pred, y, sigma_noise,device)
    n_x = torch.as_tensor(float(x.shape[0]), device=device)
    n_theta = torch.as_tensor(float(theta.shape[0]), device=device)
    log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(n_theta)
    return torch.std_mean(-log_posterior_predictive)

def avgNLL(loglikelihood, theta, X, y):
    L = loglikelihood(theta, X, y)
    M = torch.tensor(theta.shape[0]).type_as(theta)
    log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(M)
    return torch.mean(-log_posterior_predictive)