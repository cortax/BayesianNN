import numpy as np
import math
import torch
from torch import nn
from torch import functional as F


class MeanFieldVariationalDistribution(nn.Module):
    def __init__(self, nb_dim, std_init=1., sigma=1.0, device='cpu'):
        super(MeanFieldVariationalDistribution, self).__init__()
        self.device = device
        self.nb_dim = nb_dim
        self.rho = nn.Parameter(torch.log(torch.exp(sigma*torch.ones(nb_dim, device=device)) - 1), requires_grad=True)
        self.mu = nn.Parameter(std_init*torch.randn(nb_dim, device=device), requires_grad=True)
  
        
    @property
    def sigma(self):
        return self._rho_to_sigma(self.rho)
        
    def forward(self, n=1):
        sigma = self._rho_to_sigma(self.rho)
        epsilon = torch.randn(size=(n,self.nb_dim)).to(self.device)
        return self.mu + sigma *epsilon
    
    def _rho_to_sigma(self, rho):
        sigma = torch.log(torch.exp(rho)+1.)
        return sigma

    
    def log_prob(self, x):
        S = self.sigma
        mu = self.mu
        dim=self.nb_dim
        n_x=x.shape[0]
        H=S.view(dim,1,1).inverse().view(1,1,dim)
        d=((x-mu.view(1,dim))**2).view(n_x,dim)
        const=0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
        return -0.5*(H*d).sum(2).squeeze()-const


class MeanFieldVariationalMixtureDistribution(nn.Module):
    def __init__(self, nb_comp, dim,std_init=1.,sigma=1.0, device='cpu'):
        super(MeanFieldVariationalMixtureDistribution, self).__init__()
        self.nb_comp=nb_comp
        self.output_dim=dim
        self.components= nn.ModuleList([MeanFieldVariationalDistribution(dim, std_init,sigma, device) for i in range(nb_comp)]).to(device)  
        self.device = device
        
    def sample(self, n=1):
        return torch.stack([self.components[c](n) for c in range(self.nb_comp)])

    
    def forward(self, n=1):
        d = torch.distributions.multinomial.Multinomial(n, torch.ones(self.nb_comp))
        m = d.sample()
        return torch.cat([self.components[c](int(m[c])) for c in range(self.nb_comp)])
            
    def log_prob(self, x):
        lp= torch.stack([self.components[c].log_prob(x) for c in range(self.nb_comp)],dim=0)
        return torch.logsumexp(lp,dim=0)-torch.as_tensor(float(self.nb_comp)).log()

 
 


