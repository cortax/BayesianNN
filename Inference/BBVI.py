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

    def sample_parameters(self, M=1):
        size = [M]+list(self.q_weight_mu.size())
        weight_epsilon = torch.randn(size=size).to(self.device)
        size = [M]+list(self.q_bias_mu.size())
        bias_epsilon = torch.randn(size=size).to(self.device)
    
        sigma_weight = self._rho_to_sigma(self.q_weight_rho)
        sigma_bias = self._rho_to_sigma(self.q_bias_rho)
        
        self.weight_sample = weight_epsilon.mul(sigma_weight.unsqueeze(0)).add(self.q_weight_mu.unsqueeze(0).repeat(M,1,1))
        self.bias_sample = bias_epsilon.mul(sigma_bias.unsqueeze(0)).add(self.q_bias_mu.unsqueeze(0).repeat(M,1))
        
        return (self.weight_sample, self.bias_sample)

    def q_log_pdf(self, weight_sample, bias_sample):
        M = weight_sample.size(0)
        
        sigma_weight = self._rho_to_sigma(self.q_weight_rho)
        nw = torch.distributions.Normal(self.q_weight_mu.unsqueeze(0).repeat(M,1,1), sigma_weight.unsqueeze(0).repeat(M,1,1))
        
        sigma_bias = self._rho_to_sigma(self.q_bias_rho)
        nb = torch.distributions.Normal(self.q_bias_mu.unsqueeze(0).repeat(M,1), sigma_bias.unsqueeze(0).repeat(M,1))
        
        W = nw.log_prob(weight_sample).sum()
        B = nb.log_prob(bias_sample).sum()
        
        return W + B
    
    def prior_log_pdf(self, weight_sample, bias_sample):
        M = self.weight_sample.size(0)
        
        sigma_weight = self._rho_to_sigma(self.prior_weight_rho)
        nw = torch.distributions.Normal(self.prior_weight_mu.unsqueeze(0).repeat(M,1,1), sigma_weight.unsqueeze(0).repeat(M,1,1))
        
        sigma_bias = self._rho_to_sigma(self.prior_bias_rho)
        nb = torch.distributions.Normal(self.prior_bias_mu.unsqueeze(0).repeat(M,1), sigma_bias.unsqueeze(0).repeat(M,1))
        
        W = nw.log_prob(weight_sample).sum()
        B = nb.log_prob(bias_sample).sum()
        
        return W + B

    def requires_grad_rhos(self, v = False):
        self.q_weight_rho.requires_grad = v
        self.q_bias_rho.requires_grad = v
        
    def requires_grad_mus(self, v = False):
        self.q_weight_mu.requires_grad = v
        self.q_bias_mu.requires_grad = v
    
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
    
    def requires_grad_rhos(self, v = False):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].requires_grad_rhos(v)
        
    def requires_grad_mus(self, v = False):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].requires_grad_mus(v)
            
    def make_deterministic_rhos(self):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].q_weight_rho = nn.Parameter(self.registered_layers[k].q_weight_rho.new_full(self.registered_layers[k].q_weight_rho.size(), -10.0))
    
    def sample_parameters(self, M=1):
        layered_w_samples = []
        layered_bias_samples = []
        L = [self.registered_layers[k].sample_parameters(M) for k in range(len(self.registered_layers))]
        for k in range(len(self.registered_layers)):
            layered_w_samples.append(L[k][0])
            layered_bias_samples.append(L[k][1])
        return (layered_w_samples, layered_bias_samples)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_parameters(self, w_samples, b_samples):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].set_parameters(w_samples[k], b_samples[k]) 
    
    def KL_log_pdf(self, w_samples, b_samples):
        self.set_parameters(w_samples, b_samples)
        return (self.q_log_pdf(), self.prior_log_pdf())

    def q_log_pdf(self, layered_w_samples, layered_bias_samples):
        list_LQ = []
        list_LQ = [self.registered_layers[k].q_log_pdf(layered_w_samples[k], layered_bias_samples[k]) for k in range(len(self.registered_layers))]
        stack_LQ = torch.stack(list_LQ)
        return torch.sum(stack_LQ)
    
    def prior_log_pdf(self, layered_w_samples, layered_bias_samples):
        list_LP = []
        list_LP = [self.registered_layers[k].prior_log_pdf(layered_w_samples[k], layered_bias_samples[k]) for k in range(len(self.registered_layers))]
        stack_LP = torch.stack(list_LP)
        return torch.sum(stack_LP)
        
    def compute_elbo(self, x_data, y_data, n_samples_ELBO, sigma_noise, device):
        (layered_w_samples, layered_bias_samples) = self.sample_parameters(n_samples_ELBO)

        LQ = self.q_log_pdf(layered_w_samples, layered_bias_samples)
        LP = self.prior_log_pdf(layered_w_samples, layered_bias_samples)

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