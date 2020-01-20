import torch
from torch import nn
from Tools.NNtools import *
from torch.distributions.multivariate_normal import MultivariateNormal

nblayers = 1
layerwidth = 50
param_count = 151
experiment_name = 'Foong L1/W50'

class MLP(nn.Module):
    def __init__(self, nblayers, layerwidth):
        super(MLP, self).__init__()
        L = [1] + [layerwidth]*nblayers + [1]
        self.layers = nn.ModuleList()
        for k in range(len(L)-1):
            self.layers.append(nn.Linear(L[k], L[k+1]))

    def forward(self, x):
        for j in range(len(self.layers)-1):
            x = torch.tanh(self.layers[j](x))
        x = self.layers[-1](x)
        return x    

def get_training_data(device):
    training_data = torch.load('Experiments/Foong_L1W50/Data/foong_data.pt')
    x_train = training_data[0].to(device)
    y_train = training_data[1].to(device)
    y_train = y_train.unsqueeze(-1)
    return x_train, y_train

def get_validation_data(device):
    validation_data = torch.load('Experiments/Foong_L1W50/Data/foong_data_validation.pt')
    x_validation = validation_data[0].to(device)
    y_validation = validation_data[1].to(device)
    y_validation = y_validation.unsqueeze(-1)
    return x_validation, y_validation

def get_test_data(device):
    test_data = torch.load('Experiments/Foong_L1W50/Data/foong_data_test.pt')
    x_test = test_data[0].to(device)
    y_test = test_data[1].to(device)
    y_test = y_test.unsqueeze(-1)
    return x_test, y_test

def get_model(device):
    model = MLP(nblayers, layerwidth).to(device)
    flip_parameters_to_tensors(model)
    return model

def _log_norm(x, mu, std):
    return -0.5 * torch.log(2*np.pi*std**2) -(0.5 * (1/(std**2))* (x-mu)**2)

def get_logprior_fn(device):
    S = torch.eye(param_count).to(device)
    mu = torch.zeros(param_count).to(device)
    prior = MultivariateNormal(mu, scale_tril=S)
    def logprior(x):
        v = prior.log_prob(x).unsqueeze(-1)
        return v
    return logprior

def get_loglikelihood_fn(device):
    def loglikelihood(theta, model, x, y, sigma_noise):
        set_all_parameters(model, theta)
        y_pred = model(x)
        L = _log_norm(y_pred, y, torch.tensor([sigma_noise],device=device))
        return torch.sum(L).unsqueeze(-1)
    return loglikelihood

def get_logposterior_fn(device):
    logprior = get_logprior_fn(device)
    loglikelihood = get_loglikelihood_fn(device)
    def logposterior(theta, model, x, y, sigma_noise):
        return logprior(theta) + loglikelihood(theta, model, x, y, sigma_noise)
    return logposterior

def get_logposteriorpredictive_fn(device):
    def logposteriorpredictive(ensemble, model, x, y, sigma_noise):
        complogproba = []
        for theta in ensemble:
            set_all_parameters(model, theta)
            y_pred = model(x)
            complogproba.append(-torch.tensor(float(len(ensemble))).log() + _log_norm(y_pred, y, torch.tensor([sigma_noise],device=device)))
        return torch.logsumexp(torch.stack(complogproba), dim=0).sum().unsqueeze(-1)
    return logposteriorpredictive