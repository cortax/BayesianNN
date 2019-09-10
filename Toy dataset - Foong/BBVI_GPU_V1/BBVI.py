import numpy as np

import torch
from torch import nn
from torch import functional as F

from livelossplot import PlotLosses
#from torchviz import make_dot, make_dot_from_trace

#import math

import matplotlib.pyplot as plt
plt.style.use('ggplot')



def log_norm(x, mu, std):
    """
    Compute the log pdf of x under a normal distribution with mean mu and standard deviation std.
    """
    return -0.5 * torch.log(2*np.pi*std**2) -(0.5 * (1/(std**2))* (x-mu)**2)

def rho_to_sigma(rho):
    
    sigma = torch.log(1 + torch.exp(rho))
    
    return sigma


def sigma_to_rho(sigma):
    
    rho = torch.log(torch.exp(sigma) - 1)
    
    return rho


class ProbabilisticLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, device, bias=True):
        super(ProbabilisticLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.q_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.q_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.q_bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.q_bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.weight_epsilon = torch.Tensor(out_features, in_features)
        self.bias_epsilon = torch.Tensor(out_features, in_features)
        
        self.weight_sample = torch.Tensor(out_features, in_features)
        self.bias_sample = torch.Tensor(out_features, in_features)
        
        self.reset_parameters()
        
        mu = torch.tensor(0.0)
        rho = sigma_to_rho(torch.tensor(1.0))
        
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
        
        self.to(device)

    def generate_rand(self, device):
        
        self.weight_epsilon = torch.randn(size=self.q_weight_mu.size()).to(device)
        self.bias_epsilon = torch.randn(size=self.q_bias_mu.size()).to(device)
        
        return (self.weight_epsilon, self.bias_epsilon)
    
    def reparameterization(self):
        
        sigma_weight = rho_to_sigma(self.q_weight_rho)
        sigma_bias = rho_to_sigma(self.q_bias_rho)
        
        self.weight_sample = self.weight_epsilon.mul(sigma_weight).add(self.q_weight_mu)
        
        sigma_bias = rho_to_sigma(self.q_bias_rho)
        
        self.bias_sample = self.bias_epsilon.mul(sigma_bias).add(self.q_bias_mu)
        
        return (self.weight_sample, self.bias_sample)

    def q_log_pdf(self):
        sigma_weight = rho_to_sigma(self.q_weight_rho)
        nw = torch.distributions.Normal(self.q_weight_mu, sigma_weight)
        
        sigma_bias = rho_to_sigma(self.q_bias_rho)
        nb = torch.distributions.Normal(self.q_bias_mu, sigma_bias)
        
        return nw.log_prob(self.weight_sample).sum() + nb.log_prob(self.bias_sample).sum()
    
    def prior_log_pdf(self):
        sigma_weight = rho_to_sigma(self.prior_weight_rho)
        nw = torch.distributions.Normal(self.prior_weight_mu, sigma_weight)
        
        sigma_bias = rho_to_sigma(self.prior_bias_rho)
        nb = torch.distributions.Normal(self.prior_bias_mu, sigma_bias)
        
        return nw.log_prob(self.weight_sample).sum() + nb.log_prob(self.bias_sample).sum()
    
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.q_weight_mu, mean=0.0, std=5.0)
        torch.nn.init.constant_(self.q_weight_rho, -1.0)
        torch.nn.init.normal_(self.q_bias_mu, mean=0.0, std=5.0)
        torch.nn.init.constant_(self.q_bias_rho, -1.0)
       
    def forward(self, input):
        
        return torch.nn.functional.linear(input, self.weight_sample, bias=self.bias_sample)

class Model(nn.Module):
    
    def __init__(self, H, device):
        super(Model, self).__init__()
        
        self.linear1 = ProbabilisticLinear(1, H, device)
        self.linear2 = ProbabilisticLinear(H,1, device)
        
        self.registered_layers = []
        self.registered_layers.append(self.linear1)
        self.registered_layers.append(self.linear2)
        
        self.nb_parameters = self.count_parameters()
        
        self.to(device)
                
    def forward(self, x):
        
        out = x;
        
        for k in range(len(self.registered_layers)-1):
            
            out = torch.tanh(self.registered_layers[k](out))
            
        out = self.registered_layers[-1](out)
        
        return out
    
    def resample_parameters(self, device):
        
        for k in range(len(self.registered_layers)):
        
            self.registered_layers[k].generate_rand(device)
            self.registered_layers[k].reparameterization()
        
    def count_parameters(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def compute_elbo(self, x_data, y_data, n_samples_ELBO, sigma_noise, device):
        """
        Compute the elbo with the reparametrization trick.
        """

        L = []
        
        for i in range(0, n_samples_ELBO):
        
            self.resample_parameters(device)

            LQ = self.linear1.q_log_pdf() 
            LP = self.linear1.prior_log_pdf() 
    
            y_pred = self.forward(x_data)
            
            LL = log_norm(y_data, y_pred.t(), torch.tensor(sigma_noise).to(device)).sum()

            L.append(LQ-LP-LL)
            
        L = torch.stack(L)
        L = torch.mean(L)
        
        return L
    
def BBVI(data, n_neurons, n_epoch, n_iter, n_samples_ELBO, opti_params, sigma_noise, n_seed, gpu, name):
    
    device = torch.device('cuda:'+gpu if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(n_seed)
    np.random.seed(n_seed)


    x_data = data[0].to(device)
    y_data = data[1].to(device)
    
    model = Model(n_neurons, device)
    
    learning_rate, patience, factor = opti_params
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor,verbose=True)
    optimizer.zero_grad()

    liveloss = PlotLosses(fig_path='Results/BBVI_LOSS_PLOT_'+name)

    for j in range(n_epoch):
        
        logs = {}
        losses = [None] * n_iter
        
        for k in range(n_iter):
            optimizer.zero_grad()
            loss = model.compute_elbo(x_data, y_data, n_samples_ELBO, sigma_noise, device)
            losses[k] = loss
            loss.backward()
            optimizer.step()
            
        logs['expected_loss'] = torch.stack(losses).mean().detach().clone().cpu().numpy()
        logs['learning rate'] = optimizer.param_groups[0]['lr']
        liveloss.update(logs)
        liveloss.draw()
        scheduler.step(logs['expected_loss'])
        
    return model


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