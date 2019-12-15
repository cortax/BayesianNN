from torch import nn
import torch
import numpy as np
from livelossplot import PlotLosses

class PTMCMCSampler():
    def __init__(self, logposterior, theta_dim, baseMHproposalNoise=1.0, temperatureNoiseReductionFactor=1.0, temperatures=[1], device='cpu'):
        self.logposterior = logposterior
        self.theta_dim = theta_dim
        self.device = device
        self.nb_chains = len(temperatures)
        
        self.state = [[] for i in range(self.nb_chains)]
        self._swapAcceptanceCount = [0 for i in range(self.nb_chains-1)]
        self._ladderAcceptanceCount = [0 for i in range(self.nb_chains)]
        self.logProbaMatrix = [[] for i in range(self.nb_chains)]
        
        self.temperatures = temperatures
        self.baseMHproposalNoise = baseMHproposalNoise
        self.temperatureNoiseReductionFactor = temperatureNoiseReductionFactor

    def initChains(self, stateInit=None):
        if stateInit is not None:
            self.state = [[stateInit[i]] for i in range(self.nb_chains)]
        else:
            self.state = [[self._MAP()] for i in range(self.nb_chains)]
    
        self.logProbaMatrix = [ [self.logposterior(self.state[j][-1])] for j in range(self.nb_chains)]
        self._swapAcceptanceCount = [0 for i in range(self.nb_chains-1)]
        self._ladderAcceptanceCount = [0 for i in range(self.nb_chains)]
        
    def _MetropolisHastings(self, theta_current, T):
        theta_proposal = theta_current + torch.empty([1,self.theta_dim], device=self.device).normal_(std=self.baseMHproposalNoise*T**self.temperatureNoiseReductionFactor)
                   
        p_proposal = self.logposterior(theta_proposal)
        p_current = self.logposterior(theta_current)

        logA = p_proposal/T - p_current/T
        A = logA.exp()
                      
        if torch.distributions.uniform.Uniform(0.0,1.0).sample() < A.cpu():
            return theta_proposal, 1
        else:
            return theta_current, 0
                       
    def run(self, N):
        with torch.no_grad():                          
            for _ in range(N):            
                for j in range(self.nb_chains):
                    theta_current = self.state[j][-1]
                    T = self.temperatures[j]
                    theta_current, accept = self._MetropolisHastings(theta_current, T)
                    self.state[j].append(theta_current)
                    #ChainTrack[j].append(ChainTrack[j][-1])
                    if accept:
                        self._ladderAcceptanceCount[j] += 1

                for  j in range(self.nb_chains):
                    self.logProbaMatrix[j].append(self.logposterior(self.state[j][-1]))

                for j in np.random.permutation(self.nb_chains-1):
                    T_left = self.temperatures[j]
                    T_right = self.temperatures[j+1]

                    logA = self.logProbaMatrix[j][-1]/T_right + self.logProbaMatrix[j+1][-1]/T_left \
                         - self.logProbaMatrix[j][-1]/T_left - self.logProbaMatrix[j+1][-1]/T_right

                    A = torch.exp(logA).cpu()

                    if torch.distributions.uniform.Uniform(0.0,1.0).sample() < A:
                        tmp = self.state[j][-1]
                        self.state[j][-1] = self.state[j+1][-1]
                        self.state[j+1][-1] = tmp

                        #tmp = ChainTrack[j][-1]
                        #ChainTrack[j][-1] = ChainTrack[j+1][-1]
                        #ChainTrack[j+1][-1] = tmp

                        self._swapAcceptanceCount[j] = self._swapAcceptanceCount[j]+1
            
            x = self.state
            ladderAcceptanceRate = torch.tensor(self._ladderAcceptanceCount).float()/N
            swapAcceptanceRate = torch.tensor(self._swapAcceptanceCount).float()/N
            return x, ladderAcceptanceRate, swapAcceptanceRate
                      
                      
                      
    def _MAP(self, device=None):
        if device is None:
            device = self.device
            
        theta = torch.nn.Parameter( torch.empty([1,self.theta_dim],device=device).normal_(std=1.0), requires_grad=True)

        optimizer = torch.optim.Adam([theta], lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.8)
        for t in range(100000):
            optimizer.zero_grad()

            L = -torch.mean(self.logposterior(theta))
            L.backward()

            learning_rate = optimizer.param_groups[0]['lr']

            scheduler.step(L.detach().clone().cpu().numpy())
            optimizer.step()

            if learning_rate < 0.001:
                break
        return theta.detach().clone()               







                      
                      
                      