import numpy as np

import torch
from torch import nn
from torch import functional as F

from livelossplot import PlotLosses

import matplotlib.pyplot as plt
plt.style.use('ggplot')

class ProbabilisticLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, device=None, bias=True):
        super(ProbabilisticLinear, self).__init__()
        
        self.device = device
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.q_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.q_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.q_bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.q_bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.weight_epsilon = None # torch.Tensor(out_features, in_features)
        self.bias_epsilon = None #torch.Tensor(out_features, in_features)
        
        self.weight_sample = None #torch.Tensor(out_features, in_features)
        self.bias_sample = None #torch.Tensor(out_features, in_features)
        
        self.reset_parameters()
        
        mu = torch.tensor(0.0)
        sigma = torch.tensor(1.0)
        rho = torch.log(torch.exp(sigma) - 1)
        
        self.prior_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.prior_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.prior_bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.prior_bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.prior_weight_mu.requires_grad = False
        self.prior_weight_rho.requires_grad = False
        self.prior_bias_mu.requires_grad = False
        self.prior_bias_rho.requires_grad = False
        
        nn.init.constant_(self.prior_weight_mu, mu)
        nn.init.constant_(self.prior_weight_rho, rho)
        nn.init.constant_(self.prior_bias_mu, mu)
        nn.init.constant_(self.prior_bias_rho, rho)
        
        self.to(self.device)

    def _rho_to_sigma(self, rho):
        sigma = torch.log(1 + torch.exp(rho))
        return sigma

    def _sigma_to_rho(self, sigma):
        rho = torch.log(torch.exp(sigma) - 1)
        return rho

    def generate_rand(self, M=1):
        size = [M]+list(self.q_weight_mu.size())
        self.weight_epsilon = torch.randn(size=size).to(self.device)
        size = [M]+list(self.q_bias_mu.size())
        self.bias_epsilon = torch.randn(size=size).to(self.device)
    
    def reparameterization(self):
        M = self.weight_epsilon.size(0)
        sigma_weight = self._rho_to_sigma(self.q_weight_rho)
        sigma_bias = self._rho_to_sigma(self.q_bias_rho)
        
        self.weight_sample = self.weight_epsilon.mul(sigma_weight.unsqueeze(0)).add(self.q_weight_mu.unsqueeze(0).repeat(M,1,1))
        self.bias_sample = self.bias_epsilon.mul(sigma_bias.unsqueeze(0)).add(self.q_bias_mu.unsqueeze(0).repeat(M,1))

    def q_log_pdf(self):
        M = self.weight_sample.size(0)
        
        sigma_weight = self._rho_to_sigma(self.q_weight_rho)
        nw = torch.distributions.Normal(self.q_weight_mu.unsqueeze(0).repeat(M,1,1), sigma_weight.unsqueeze(0).repeat(M,1,1))
        
        sigma_bias = self._rho_to_sigma(self.q_bias_rho)
        nb = torch.distributions.Normal(self.q_bias_mu.unsqueeze(0).repeat(M,1), sigma_bias.unsqueeze(0).repeat(M,1))
        
        W = nw.log_prob(self.weight_sample).sum()
        B = nb.log_prob(self.bias_sample).sum()
        
        return W + B
    
    def prior_log_pdf(self):
        M = self.weight_sample.size(0)
        
        sigma_weight = self._rho_to_sigma(self.prior_weight_rho)
        nw = torch.distributions.Normal(self.prior_weight_mu.unsqueeze(0).repeat(M,1,1), sigma_weight.unsqueeze(0).repeat(M,1,1))
        
        sigma_bias = self._rho_to_sigma(self.prior_bias_rho)
        nb = torch.distributions.Normal(self.prior_bias_mu.unsqueeze(0).repeat(M,1), sigma_bias.unsqueeze(0).repeat(M,1))
        
        W = nw.log_prob(self.weight_sample).sum()
        B = nb.log_prob(self.bias_sample).sum()
        
        return W + B

    #def set_parameters(self, weight_sample, bias_sample):
    #    #todo: Make sure this affectation does NOT carry the whole graph that generated it
    #    print('warning: set_parameters()')
    #    self.weight_sample = weight_sample
    #    self.bias_sample = bias_sample
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.q_weight_mu, mean=0.0, std=5.0)
        torch.nn.init.constant_(self.q_weight_rho, -1.0)
        torch.nn.init.normal_(self.q_bias_mu, mean=0.0, std=5.0)
        torch.nn.init.constant_(self.q_bias_rho, -1.0)
       
    def forward(self, input):
        M = self.weight_sample.size(0)
        if input.dim() < 3:
            input = input.unsqueeze(0).repeat(M,1,1)
        return input.matmul(self.weight_sample.permute(0,2,1)).add(self.bias_sample.unsqueeze(-1).permute(0,2,1))

    
class VariationalNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_width, nb_layers, device=None):
        super(VariationalNetwork, self).__init__()
        
        self.device = device
        self.registered_layers = []

        # Dynamically creating layers and registering them as parameters to optimize 
        layer_dims = [input_size] + [layer_width] * (nb_layers-1) + [output_size]
        for k in range(nb_layers):
            a = layer_dims[k]
            b = layer_dims[k+1]
            layer = ProbabilisticLinear(a, b, device)
            exec('self.linear' + str(k+1) + ' = layer')
            self.registered_layers.append(layer)
        
        self.nb_parameters = self.count_parameters()
        self.to(device)
        
    def _log_norm(self, x, mu, std):
        return -0.5 * torch.log(2*np.pi*std**2) -(0.5 * (1/(std**2))* (x-mu)**2)
                
    def forward(self, x):
        out = x;
        for k in range(len(self.registered_layers)-1):
            out = torch.tanh(self.registered_layers[k](out))
        out = self.registered_layers[-1](out)
        return out
    
    def resample_parameters(self, M=1):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].generate_rand(M)
            self.registered_layers[k].reparameterization()
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_parameters(self, w_samples, b_samples):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].set_parameters(w_samples[k], b_samples[k]) 
   
    def q_log_pdf(self):
        list_LQ = []
        for k in range(len(self.registered_layers)):
            list_LQ.append(self.registered_layers[k].q_log_pdf())
        stack_LQ = torch.stack(list_LQ)
        return torch.sum(stack_LQ)
    
    def prior_log_pdf(self):
        list_LP = []
        for k in range(len(self.registered_layers)):
            list_LP.append(self.registered_layers[k].prior_log_pdf())
        stack_LP = torch.stack(list_LP)
        return torch.sum(stack_LP)
        
    def compute_elbo(self, x_data, y_data, n_samples_ELBO, sigma_noise, device):
        self.resample_parameters(n_samples_ELBO)

        LQ = self.q_log_pdf()
        LP = self.prior_log_pdf() 

        y_pred = self.forward(x_data)
        LL = self._log_norm(y_pred, y_data, torch.tensor(sigma_noise).to(device))

        L = (LQ - LP - LL.sum())
        
        return torch.div(L, n_samples_ELBO)

class VariationalOptimizer():
    def __init__(self, model, sigma_noise, optimizer, optimizer_params, scheduler=None, scheduler_params=None):
        self.model = model
        self.sigma_noise = sigma_noise
        self.device = model.device
        
        self.optimizer = optimizer(model.parameters(), **optimizer_params)
        if scheduler is None:           
            self.scheduler = None
        else:
            self.scheduler = scheduler(self.optimizer, **scheduler_params)
            
    def run(self, data, n_epoch=100, n_iter=1, n_ELBO_samples=1, seed=None, plot=False, savePath=None, saveName=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x_data = data[0].to(self.device)
        y_data = data[1].to(self.device)

        self.optimizer.zero_grad()

        if saveName is not None and savePath is not None:
            liveloss = PlotLosses(fig_path=str(savePath)+str(saveName))
        else:
            liveloss = PlotLosses()

        for j in range(n_epoch):
            logs = {}
            losses = [None] * n_iter

            for k in range(n_iter):
                self.optimizer.zero_grad()
                loss = self.model.compute_elbo(x_data, y_data, n_ELBO_samples, self.sigma_noise, self.device)
                losses[k] = loss
                loss.backward()
                self.optimizer.step()

            logs['expected_loss'] = torch.stack(losses).mean().detach().clone().cpu().numpy()
            logs['learning rate'] = self.optimizer.param_groups[0]['lr']
            liveloss.update(logs)
            if plot is True:
                liveloss.draw()
            if self.scheduler is not None:
                self.scheduler.step(logs['expected_loss'])
        return self.model

def plot_BBVI_Uncertainty(model, data, name, gpu):
    
    device = torch.device('cuda:'+gpu if torch.cuda.is_available() else 'cpu')
    x_test = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)

    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)

    plt.scatter(data[0].cpu(), data[1].cpu())
    plt.axis([-2, 2, -2, 6])

    for _ in range(1000): 

        model.linear1.generate_rand(device)
        model.linear1.reparameterization()

        y_test = model.forward(x_test)
        
        plt.plot(x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), alpha=0.05, linewidth=1,color='lightblue')
        
    plt.savefig('Results/BBVI_PLOT_'+name)