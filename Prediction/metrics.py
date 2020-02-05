import torch
import math

def _log_norm(x, mu, std,device):
    """
    Evaluation of 1D normal distribution on tensors

    Parameters:
        x (Tensor): Data tensor of size B X S 
        mu (Tensor): Mean tensor of size S 
        std (Float): Positive scalar (standard deviation)

    Returns:
        logproba (Tensor): Same size as x with logproba(b,i)=log p(x(b,i),mu(i),std)
    """
    B=x.shape[0]
    S=x.shape[1]
    var=torch.as_tensor(std**2,device=device)
    d=(x.view(B,S,1)-mu.view(1,S,1))**2
    c=2*math.pi*var
    return -0.5 * (1/(var))*d -0.5 * c.log()


def logprior(x,device):
    dim=x.shape[-1]
    S = torch.ones(dim).to(device)
    mu = torch.zeros(dim).to(device)
    n_x=x.shape[0]

    H=S.view(dim,1,1).inverse().view(1,1,dim)
    d=((x-mu.view(1,dim))**2).view(n_x,dim)
    const=0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
    return -0.5*(H*d).sum(2).squeeze()-const
#test!
#prior=torch.distributions.multivariate_normal.MultivariateNormal(mu,covariance_matrix=torch.diagflat(S))
#x=torch.rand(10,param_count).to(device)
#torch.allclose(logprior(x),prior.log_prob(x))


def loglikelihood(theta, model,x, y, sigma_noise,device):
    y_pred = model(x, theta)
    L=_log_norm(y_pred,y,sigma_noise,device)
    return torch.sum(L, dim=1).squeeze()
    
def logposterior(theta,model, x, y, sigma_noise,device):
    return logprior(theta,device)+loglikelihood(theta,model, x, y, sigma_noise,device)

def get_logposterior(model,x,y,sigma_noise,device):
    def logtarget(theta):
        return logposterior(theta,model, x,y,sigma_noise,device)
    return logtarget

##Metrics

#NLPD from Quinonero-Candela and al.
# the average negative log predictive density (NLPD) of the true targets

def NLPD(theta,model, x, y, sigma_noise,inv_transform,device):
    y_pred =inv_transform(model(x, theta))
    L = _log_norm(y_pred, y, sigma_noise,device)
    n_x = torch.as_tensor(float(x.shape[0]), device=device)
    n_theta = torch.as_tensor(float(theta.shape[0]), device=device)
    log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(n_theta)
    return torch.std_mean(-log_posterior_predictive)

#root mean square error

def RMSE(theta,model,x,y,inv_transform,device):
    n_samples=x.shape[0]
    y_pred =inv_transform(model(x, theta)).mean(0)
    mse=(y_pred-y.view(n_samples,1))**2
    return mse.mean().sqrt()

