import numpy as np
import sys
import os
import math
import torch
from torch import nn
from torch import functional as F
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
from Inference.Variational import MeanFieldVariationalDistribution
from Inference.VariationalBoosting import MeanFieldVariationalMixtureDistribution
from Tools.NNtools import *
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

def main(tag, nbMAP, nblayers, layerwidth, device):
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
    MAPS = []
    
    for _ in range(nbMAP):
        std = torch.distributions.Gamma(torch.tensor([1.0]), torch.tensor([1.0])).sample()[0].float()
        theta = torch.nn.Parameter( torch.empty([1,param_count],device=device).normal_(std=std), requires_grad=True)

        optimizer = torch.optim.Adam([theta], lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, factor=0.9)
        for t in range(1000000):
            optimizer.zero_grad()

            L = -torch.mean(logtarget(theta))
            L.backward()

            learning_rate = optimizer.param_groups[0]['lr']

            scheduler.step(L.detach().clone().cpu().numpy())
            optimizer.step()

            if learning_rate < 0.0005:
                break
        
        MAPS.append([theta.detach().clone()])
        
        filename = "Results/"+'L'+str(nblayers)+'W'+str(layerwidth)+"/MAP/MAP_batch_" + str(tag) + ".pt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        pickle_out = open(filename,"wb")
        pickle.dump(MAPS, pickle_out)
        pickle_out.close()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layerwidth", type=int,
                        help="number of neurones per layer")
    parser.add_argument("--nblayers", type=int,
                        help="number of layers")
    parser.add_argument("--tag", type=int,
                        help="identifier for the learning run")
    parser.add_argument("--nbMAP", type=int,
                        help="number of MAPs to compute")
    parser.add_argument("--device", type=str,
                        help="force device to be used")
    args = parser.parse_args()
    print(args)

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    main(args.tag, args.nbMAP, args.nblayers, args.layerwidth, device)
