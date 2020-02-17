from torch import nn
import torch
import numpy as np

from Inference.PointEstimate import AdamGradientDescent


class PTMCMCSampler():
    def __init__(self, logposterior, theta_dim, baseMHproposalNoise=1.0, temperatureNoiseReductionFactor=1.0, temperatures=[1], device='cpu'):
        self.logposterior = logposterior
        self.theta_dim = theta_dim
        self.device = device
        self.nb_chains = len(temperatures)
        
        self.state = [[] for i in range(self.nb_chains)]
        self.current= [0 for i in range(self.nb_chains)]
        self.last= [0 for i in range(self.nb_chains)]

        self._swapAcceptanceCount = [0 for i in range(self.nb_chains-1)]
        self._ladderAcceptanceCount = [0 for i in range(self.nb_chains)]
 #       self.logProbaMatrix = [[] for i in range(self.nb_chains)]
        
        self.temperatures = temperatures
        self.baseMHproposalNoise = baseMHproposalNoise
        self.temperatureNoiseReductionFactor = temperatureNoiseReductionFactor

    def initChains(self, nbiter=1000, std_init=1.0, stateInit=None):
        if stateInit is not None:
            self.last = [stateInit[i] for i in range(self.nb_chains)]
        else:
            self.last = [self._MAP(nbiter, std_init) for _ in range(self.nb_chains)]
    
        self.logProbaMatrix = [ [self.logposterior(self.last[j])] for j in range(self.nb_chains)]
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
                       
    def run(self, N, burnin, thinning):
        with torch.no_grad():                          
            for t in range(N):            
                for j in range(self.nb_chains):
                    theta_last = self.last[j]
                    T = self.temperatures[j]
                    theta_current, accept = self._MetropolisHastings(theta_last, T)
                    self.current[j]=theta_current
                    #ChainTrack[j].append(ChainTrack[j][-1])
                    if accept:
                        self._ladderAcceptanceCount[j] += 1


                for  j in range(self.nb_chains):
                    self.logProbaMatrix[j]=self.logposterior(self.current[j]) # append new_state

                if t % 100 ==0:
                    ladderAcceptanceRate = torch.as_tensor(self._ladderAcceptanceCount).float() / (t+1)
                    swapAcceptanceRate = torch.as_tensor(self._swapAcceptanceCount).float()/ (t+1)
                    ladderAcceptanceRate= ladderAcceptanceRate.tolist()
                    swapAcceptanceRate=swapAcceptanceRate.tolist()
                    stats = 'Epoch [{0}/{0}]'.format(t+1, N )+'Acceptance: '+str(ladderAcceptanceRate)+ 'Swap: '+str(swapAcceptanceRate)
                    print(stats)

                for j in np.random.permutation(self.nb_chains-1):
                    T_left = self.temperatures[j]
                    T_right = self.temperatures[j+1]

                    logA = self.logProbaMatrix[j]/T_right + self.logProbaMatrix[j+1]/T_left \
                         - self.logProbaMatrix[j]/T_left - self.logProbaMatrix[j+1]/T_right

                    A = torch.exp(logA).cpu()

                    if torch.distributions.uniform.Uniform(0.0,1.0).sample() < A:
                        tmp = self.current[j] #swap
                        self.current[j] = self.current[j+1]
                        self.current[j+1] = tmp #swap

                        self._swapAcceptanceCount[j] = self._swapAcceptanceCount[j]+1

                self.last=self.current
                if (t - burnin) % thinning == 0:
                    for j in range(self.nb_chains):
                        self.state[j].append(self.current[j])  # append new_state

            x = self.state
#            logProba = self.logProbaMatrix
            ladderAcceptanceRate = torch.as_tensor(self._ladderAcceptanceCount).float()/N
            swapAcceptanceRate = torch.as_tensor(self._swapAcceptanceCount).float()/N

            return x, ladderAcceptanceRate, swapAcceptanceRate #, logProba

    def _MAP(self, nbiter, std_init,device='cpu'):
        optimizer = AdamGradientDescent(self.logposterior, nbiter, .01, .00000001, 50, .5, device, True)

        theta0 = torch.empty((1, self.theta_dim), device=device).normal_(0., std=std_init)
        best_theta, best_score, score = optimizer.run(theta0)

        return best_theta.detach().clone()
    # def _MAP(self, nbiter, std_init, device=None):
    #     if device is None:
    #         device = self.device
    #     theta = torch.nn.Parameter( torch.empty([1,self.theta_dim],device=device).normal_(std=std_init), requires_grad=True)
    #
    #     optimizer = torch.optim.Adam([theta], lr=0.01)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    #     for t in range(nbiter):
    #         optimizer.zero_grad()
    #
    #         L = -torch.mean(self.logposterior(theta))
    #         L.backward()
    #
    #         learning_rate = optimizer.param_groups[0]['lr']
    #
    #         scheduler.step(L.detach().clone().cpu().numpy())
    #         optimizer.step()
    #
    #         if learning_rate < 0.0001:
    #             break
    #     return theta.detach().clone()
