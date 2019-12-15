import sys
import os
from os.path import dirname

import numpy as np
import math
import torch
from torch import nn
from torch import functional as F
import scipy.stats as stats
from Inference.Variational import MeanFieldVariationalDistribution
from Tools.NNtools import *
from Inference.ParallelTempering import *

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

data = torch.load('Data/foong_data.pt')
x_data = data[0].to(device)
y_data = data[1].to(device)
y_data = y_data.unsqueeze(-1)

model = nn.Sequential( nn.Linear(1, 20),
                       nn.Tanh(), 
                       nn.Linear(20, 1),
                     ).to(device)
param_count = get_param(model).shape[0]
flip_parameters_to_tensors(model)


prior = MeanFieldVariationalDistribution(param_count, sigma=1.0, device=device)
prior.mu.requires_grad = False
prior.rho.requires_grad = False
def logprior(x):
    return prior.log_prob(x)

def loglikelihood(theta, model, x, y, sigma_noise):
    def _log_norm(x, mu, std):
        return -0.5 * torch.log(2*np.pi*std**2) -(0.5 * (1/(std**2))* (x-mu)**2)
    set_all_parameters(model, theta)
    y_pred = model(x)
    L = _log_norm(y_pred, y, torch.tensor([sigma_noise],device=device))
    return torch.sum(L).unsqueeze(-1)

def logposterior(theta, model, x, y, sigma_noise):
    return logprior(theta) + loglikelihood(theta, model, x, y, sigma_noise)

N = 100000
temperatures = [0.5, 1, 1.5, 2, 3, 5, 10, 20, 40, 120, 500]
stateInit = [prior.sample() for i in range(len(temperatures))]
baseMHproposalNoise = 0.003
temperatureNoiseReductionFactor = 0.5

logtarget = lambda theta : logposterior(theta, model, x_data, y_data, 0.1 )

sampler = PTMCMCSampler(logtarget, param_count, baseMHproposalNoise, temperatureNoiseReductionFactor, temperatures, device)

import pickle
state = pickle.load( open( "MAPS.pt", "rb" ) )
sampler.initChains([state[i][0] for i in range(len(state))])

#sampler.initChains()

for s in range(10):
    print(s)
    x, ladderAcceptanceRate, swapAcceptanceRate = sampler.run(1000)
    pickle_out = open("PTMCMC_checkpoint.pt","wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()