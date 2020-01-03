import numpy as np
import math
import torch
from torch import nn
from torch import functional as F


from livelossplot import PlotLosses

import matplotlib.pyplot as plt
plt.style.use('ggplot')


class MeanFieldVariationalMixtureDistribution(nn.Module):
    def __init__(self, proportions, components, device='cpu'):
        self.proportions = proportions.to(device)
        self.components = components
        self.requires_grad_(False)
        self.device = device
        
    def sample(self, n=1):
        d = torch.distributions.multinomial.Multinomial(n, self.proportions)
        m = d.sample()
        return torch.cat([self.components[c].sample(int(m[c])) for c in range(len(self.components))])
        
    def requires_grad_(self, b):
        self.proportions.requires_grad_(b) 
        for c in self.components:
            c.requires_grad_(b)
            
    def log_prob(self, x):
        #return torch.logsumexp(torch.stack([torch.log(self.proportions[c]) + self.components[c].log_prob(x) for c in range(len(self.proportions))],dim=1),dim=1)
        MU = torch.cat([c.mu for c in self.components], dim=0)
        SIGMA = torch.stack([c.sigma for c in self.components])
        P = -0.5*torch.log(2*np.pi*SIGMA**2) - ( 0.5*(MU-x)**2)/(SIGMA**2) 
        return torch.logsumexp(torch.log(self.proportions) + P.sum(dim=1), dim=0)
        
    def log_prob_augmented(self, x, q_new, unbounded_prop_new):
        prop_new = torch.sigmoid(unbounded_prop_new).to(self.device)
        A = torch.log(1-prop_new) + self.log_prob(x)
        B = torch.log(prop_new) + q_new.log_prob(x)
        return torch.logsumexp(torch.stack([A,B]),dim=0)


class VariationalBoostingOptimizer():
    def __init__(self, mixture, sigma_noise, optimizer, optimizer_params, scheduler=None, scheduler_params=None, min_lr=None):
        self.mixture = mixture
        self.sigma_noise = sigma_noise
        self.device = mixture.device
        self.min_lr = min_lr
        
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        if scheduler is None:           
            self.scheduler = None
            self.scheduler_params = None
        else:
            self.scheduler = scheduler
            self.scheduler_params = scheduler_params
            
    def run(self, data, nb_component=1, n_epoch=100, n_iter=1, n_ELBO_samples=1, seed=None, plot=False, log=False, verbose=0):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x_data = data[0].to(self.device)
        y_data = data[1].to(self.device)
        
        for c in range(nb_component):
            if len(self.mixture.components) == 0:
                new_component = VariationalNetwork(self.mixture.input_size, self.mixture.output_size, \
                                               self.mixture.layer_width, self.mixture.nb_layers, \
                                               device=self.mixture.device)

                voptimizer = BBVI.VariationalOptimizer(model=new_component, sigma_noise=self.sigma_noise, \
                             optimizer=self.optimizer, optimizer_params=self.optimizer_params, \
                             scheduler=self.scheduler, scheduler_params=self.scheduler_params, min_lr=self.min_lr)
                new_component = voptimizer.run((x_data,y_data), n_epoch=n_epoch, n_iter=n_iter, n_ELBO_samples=n_ELBO_samples, plot=plot)
                #epoch_count = voptimizer.last_epoch
                self.mixture.add_component(new_component, torch.tensor(1.0, device=self.mixture.device))
            else:
                new_component = VariationalNetwork(self.mixture.input_size, self.mixture.output_size, \
                                                   self.mixture.layer_width, self.mixture.nb_layers, \
                                                   device=self.mixture.device)

                new_unscaled_proportion = nn.Parameter(torch.tensor(0.0, requires_grad=False, device=self.device), requires_grad = False)

                parameters = list(new_component.parameters()) + [new_unscaled_proportion]
                vo = self.optimizer(parameters, **self.optimizer_params)

                if self.scheduler is not None:
                    s = self.scheduler(vo, **self.scheduler_params)

                vo.zero_grad()

                if plot:
                    liveloss = PlotLosses()

                for j in range(n_epoch):
                    logs = {}
                    losses = [None] * n_iter

                    for k in range(n_iter):
                        vo.zero_grad()
                        new_proportion = torch.sigmoid(new_unscaled_proportion)
                        loss = self.mixture.compute_elbo(x_data, y_data, n_samples_ELBO=n_ELBO_samples, sigma_noise=self.sigma_noise, new_component=new_component, new_proportion=new_proportion)
                        losses[k] = loss
                        loss.backward()
                        vo.step()
                        
                    expected_loss = torch.stack(losses).mean().detach().clone().cpu().numpy()
                    learning_rate = vo.param_groups[0]['lr']

                    if plot is True:
                        logs['expected_loss'] = expected_loss
                        logs['learning rate'] = learning_rate
                        liveloss.update(logs)
                        liveloss.draw()
                        
                    if verbose >= 1:
                        print("Epoch: {0:6d} / training loss: {1:16.3f} / learning rate: {2:1.7f}".format(j,expected_loss,learning_rate))    
                        
                    if log:
                        self.log.append("Epoch: {0:6d} / training loss: {1:16.3f} / learning rate: {2:1.7f}".format(j,expected_loss,learning_rate))
                    
                    if s is not None:
                        s.step(logs['expected_loss'])
                        
                    if self.scheduler is not None:
                        s.step(expected_loss)
                        if self.min_lr is not None and learning_rate < self.min_lr:
                            new_proportion = torch.sigmoid(new_unscaled_proportion)
                            self.mixture.add_component(new_component, torch.tensor(new_proportion, device=self.mixture.device))
                            return self.mixture
                new_proportion = torch.sigmoid(new_unscaled_proportion)
                self.mixture.add_component(new_component, new_proportion)
        return self.mixture    

