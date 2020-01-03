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
import pickle
import argparse

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

def main(tag, nbBBVI, nblayers, layerwidth, device):
    data = torch.load('Data/foong_data.pt')
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = y_data.unsqueeze(-1)

    model = MLP(nblayers, layerwidth).to(device)
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

    
    logtarget = lambda theta : logposterior(theta, model, x_data, y_data, 0.1 )
    BBVI = []

    #dirname = "Results/"+'L'+str(nblayers)+'W'+str(layerwidth)+"/PTMCMC_hot/"
    #os.makedirs(dirname, exist_ok=True)

    for _ in range(nbBBVI):
        std = torch.distributions.Gamma(torch.tensor([1.0]), torch.tensor([1.0])).sample()[0].float()
        q = MeanFieldVariationalDistribution(param_count, sigma=0.0000001, device=device)
        q.mu = nn.Parameter( torch.empty([1,param_count],device=device).normal_(std=std), requires_grad=True)
        q.rho.requires_grad = False
        q.mu.requires_grad = True
        
        n_samples_ELBO = 10
        optimizer = torch.optim.Adam(q.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.9)

        for _ in range(1000000):
            optimizer.zero_grad()

            listDIV = []
            for i in range(n_samples_ELBO):
                z = q.sample(1)
                LQ = q.log_prob(z)
                LP = logposterior(z, model, x_data, y_data, sigma_noise=0.1)
                listDIV.append((LQ - LP))

            L = torch.stack(listDIV).mean()
            L.backward()

            learning_rate = optimizer.param_groups[0]['lr']

            scheduler.step(L.detach().clone().cpu().numpy())
            optimizer.step()

            if learning_rate < 0.00001:
                break

        n_samples_ELBO = 100
        optimizer = torch.optim.Adam(q.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.9)
        
        q.mu.detach_().requires_grad_(False)
        q.rho.detach_().requires_grad_(True)

        for _ in range(1000000):
            optimizer.zero_grad()

            listDIV = []
            for i in range(n_samples_ELBO):
                z = q.sample(1)
                LQ = q.log_prob(z)
                LP = logposterior(z, model, x_data, y_data, sigma_noise=0.1)
                listDIV.append((LQ - LP))

            L = torch.stack(listDIV).mean()
            L.backward()

            learning_rate = optimizer.param_groups[0]['lr']

            scheduler.step(L.detach().clone().cpu().numpy())
            optimizer.step()

            if learning_rate < 0.00001:
                break

        BBVI.append(q)
        filename = "Results/"+'L'+str(nblayers)+'W'+str(layerwidth)+"/BBVI_expansion/BBVI_batch_" + str(tag) + ".pt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        pickle_out = open(filename,"wb")
        pickle.dump(BBVI, pickle_out)
        pickle_out.close()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbBBVI", type=int,
                        help="number of BBVIs to compute")
    parser.add_argument("--layerwidth", type=int,
                        help="number of neurones per layer")
    parser.add_argument("--nblayers", type=int,
                        help="number of layers")
    parser.add_argument("--tag", type=int,
                        help="identifier for the learning run")
    parser.add_argument("--device", type=str,
                        help="force device to be used")
    args = parser.parse_args()
    print(args)

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    main(args.tag, args.nbBBVI, args.nblayers, args.layerwidth, device)

