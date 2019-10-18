import numpy as np
import math
import torch
from torch import nn
from torch import functional as F
from Inference.BBVI import VariationalNetwork
from Inference import BBVI

from livelossplot import PlotLosses

import matplotlib.pyplot as plt
plt.style.use('ggplot')


class MixtureVariationalNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_width, nb_layers, activation=torch.tanh, device=None):
        super(MixtureVariationalNetwork, self).__init__()
        
        self.activation = activation
        self.layer_width = layer_width
        self.input_size = input_size
        self.output_size = output_size
        self.nb_layers = nb_layers
        self.device = device
        
        self.components = []
        self.pi = torch.tensor([], device=device)
        
    def add_component(self, component, proportion):
        #todo check compatibility with other components
        component.requires_grad_rhos(False)
        component.requires_grad_mus(False)
        self.components.append(component)
        proportion = proportion.to(self.device)
        self.pi = torch.cat((self.pi*(1-proportion), proportion.unsqueeze(0)))
        self.pi.require_grad = False
        
    def sample_parameters(self, M=1, new_component=None, new_proportion=torch.tensor(0.0)):
        npi = torch.tensor([1.0-new_proportion, new_proportion], device=self.device)
        D = torch.distributions.multinomial.Multinomial(M, npi)
        m_oldnew = D.sample()
        if int(m_oldnew[0]) > 0:
            D_old = torch.distributions.multinomial.Multinomial(int(m_oldnew[0]), self.pi)
            m_old = D_old.sample()
        else:
            m_old = torch.zeros(self.pi.size())

        S = []    
        for j in range(len(self.pi)):
            t = self.components[j].sample_parameters(int(m_old[j]))
            if int(m_old[j]) > 0: 
                S.append(t)
        if new_component is not None:
            S.append(new_component.sample_parameters(int(m_oldnew[1])))

        return ([torch.cat([c[0][k] for c in S]) for k in range(self.nb_layers)], [torch.cat([c[1][k] for c in S]) for k in range(self.nb_layers)])

    def q_log_pdf(self, layered_w_samples, layered_bias_samples):
        log_q = [c.q_log_pdf(layered_w_samples, layered_bias_samples) for c in self.components]
        return torch.logsumexp(torch.stack(log_q) + torch.log(self.pi).unsqueeze(0).t(), dim=0)

    def prior_log_pdf(self, layered_w_samples, layered_bias_samples):
        log_prior = [c.prior_log_pdf(layered_w_samples, layered_bias_samples) for c in self.components]
        return torch.logsumexp(torch.stack(log_prior) + torch.log(self.pi).unsqueeze(0).t(), dim=0)
           
    def compute_elbo(self, x_data, y_data, n_samples_ELBO, sigma_noise, new_component=None, new_proportion=None):
        # sample X^(c)
        (layered_w_samples_XC, layered_bias_samples_XC) = self.sample_parameters(n_samples_ELBO)

        LP_XC = self.prior_log_pdf(layered_w_samples_XC, layered_bias_samples_XC)
        y_pred_XC = self.forward(x_data)
        LL_XC = self._log_norm(y_pred_XC, y_data, torch.tensor(sigma_noise).to(self.device))
        posterior_XC = LL_XC.sum(dim=[1,2]) + LP_XC

        qC_log_XC = self.q_log_pdf(layered_w_samples_XC, layered_bias_samples_XC)

        qN_log_XC = new_component.q_log_pdf(layered_w_samples_XC, layered_bias_samples_XC)

        qCN_log_XC = torch.logsumexp(torch.stack([torch.log(torch.tensor(1.0, device=self.device)-new_proportion) + qC_log_XC, torch.log(new_proportion) + qN_log_XC],dim=0),dim=0)

        # sample X_(c+1)
        (layered_w_samples_XN, layered_bias_samples_XN) = new_component.sample_parameters(n_samples_ELBO)

        LP_XN = new_component.prior_log_pdf(layered_w_samples_XN, layered_bias_samples_XN)
        y_pred_XN = new_component.forward(x_data)
        LL_XN = new_component._log_norm(y_pred_XN, y_data, torch.tensor(sigma_noise).to(self.device))
        posterior_XN = LL_XN.sum(dim=[1,2]) + LP_XN

        qC_log_XN = self.q_log_pdf(layered_w_samples_XN, layered_bias_samples_XN)

        qN_log_XN = new_component.q_log_pdf(layered_w_samples_XN, layered_bias_samples_XN)
        qCN_log_XN = torch.logsumexp(torch.stack([torch.log(torch.tensor(1.0, device=self.device)-new_proportion) + qC_log_XN, \
                torch.log(new_proportion) + qN_log_XN],dim=0),dim=0)

        L = (torch.tensor(1.0, device=self.device)-new_proportion) * (posterior_XC.mean() - qCN_log_XC.mean()) + new_proportion * (posterior_XN.mean() - qCN_log_XN.mean()) 
        #L = new_proportion*(posterior_XN.mean() - qCN_log_XN.mean())
        return -L
    
    def requires_grad_rhos(self, v = False):
        for k in range(len(self.components)):
            self.components[k].requires_grad_rhos(v)
        
    def requires_grad_mus(self, v = False):
        for k in range(len(self.components)):
            self.components[k].requires_grad_mus(v)
    
    def forward(self, x):
        return torch.cat([self.components[k].forward(x) for k in range(len(self.components))],dim=0)
    

    def _log_norm(self, x, mu, std):
        return -0.5 * torch.log(2*np.pi*std**2) -(0.5 * (1/(std**2))* (x-mu)**2)
    
    
    def get_mixture(self):
        mixture = {}
        
        mixture['activation'] = self.activation
        mixture['input_size'] = self.input_size
        mixture['output_size'] = self.output_size
        mixture['layer_width'] = self.layer_width
        mixture['nb_layers'] = self.nb_layers
        
        components = []
        for c in self.components:
            components.append(c.get_network())
            
        mixture['components'] = components
        mixture['pi'] = self.pi.to('cpu')

        return mixture
    
    
    def set_mixture(self, mixture):
        self.activation = mixture['activation']
        self.input_size = mixture['input_size'] 
        self.output_size = mixture['output_size']
        self.layer_width = mixture['layer_width']
        self.nb_layers = mixture['nb_layers']

        for i, newparam in enumerate(mixture['components']):
            Net = VariationalNetwork(self.input_size, self.output_size, self.layer_width, self.nb_layers, self.activation)
            Net.set_network(newparam)
            self.add_component(Net, torch.tensor(0.5))
            
        self.pi = mixture['pi']

            
    def set_device(self, device):
        for c in self.components:
            c.set_device(device)
    
    
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
    
