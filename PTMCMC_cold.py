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

def main(tag, nb_chunk, chunk_size, nblayers, layerwidth, device):
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

    
    temperatures = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    stateInit = [prior.sample() for i in range(len(temperatures))]
    baseMHproposalNoise = 0.003
    temperatureNoiseReductionFactor = 0.5

    logtarget = lambda theta : logposterior(theta, model, x_data, y_data, 0.1 )

    sampler = PTMCMCSampler(logtarget, param_count, baseMHproposalNoise, temperatureNoiseReductionFactor, temperatures, device)

    sampler.initChains()
    
    dirname = "Results/"+'L'+str(nblayers)+'W'+str(layerwidth)+"/PTMCMC_cold/"
    os.makedirs(dirname, exist_ok=True)

    for s in range(nb_chunk):
        print(s)
        if os.path.exists(dirname+"PTMCMC_checkpoint_xp_" + str(tag) + "_chunk_" + str(s) + ".pt"):
            (x, _, _, _) = pickle.load( open(dirname+"PTMCMC_checkpoint_xp_" + str(tag) + "_chunk_" + str(s) + ".pt", "rb" ) )
        else:
            x, ladderAcceptanceRate, swapAcceptanceRate, logProba = sampler.run(chunk_size)
            pickle_out = open(dirname+"PTMCMC_checkpoint_xp_" + str(tag) + "_chunk_" + str(s) + ".pt","wb")
            pickle.dump((x, ladderAcceptanceRate, swapAcceptanceRate, logProba), pickle_out)
            pickle_out.close()
        
        sampler = PTMCMCSampler(logtarget, param_count, baseMHproposalNoise, temperatureNoiseReductionFactor, temperatures, device)
        sampler.initChains([x[i][-1] for i in range(len(x))])


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbchunk", type=int,
                        help="number of saving checkpoints")
    parser.add_argument("--chunksize", type=int,
                        help="length of runs")
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

    main(args.tag, args.nbchunk, args.chunksize, args.nblayers, args.layerwidth, device)