from torch import nn
import torch
#from livelossplot import PlotLosses

class MeanFieldVariationalDistribution(nn.Module):
    def __init__(self, nb_dim, mu=0.0, sigma=1.0, device='cpu'):
        super(MeanFieldVariationalDistribution, self).__init__()
        self.device = device
        self.nb_dim = nb_dim
        self.mu = nn.Parameter(torch.Tensor(nb_dim).to(self.device), requires_grad=True)
        self.rho = nn.Parameter(torch.Tensor(nb_dim).to(self.device), requires_grad=True)
        
        if not torch.is_tensor(mu):
            mu = torch.tensor(mu)
            
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma)
        
        rho = torch.log(torch.exp(sigma) - 1)
        
        nn.init.constant_(self.mu, mu)
        nn.init.constant_(self.rho, rho)
        
    def set_mu(self, mu):
        if not torch.is_tensor(mu):
            mu = torch.tensor(mu).float()
        nn.init.constant_(self.mu, mu)
        
        
    def set_rho(self, rho):
        if not torch.is_tensor(rho):
            rho = torch.tensor(rho).float()
        nn.init.constant_(self.rho, rho)
       
    def set_sigma(self, sigma):
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma).float()
        rho = self._sigma_to_rho(sigma)
        self.set_rho(rho)
        
    @property
    def sigma(self):
        return self._rho_to_sigma(self.rho)
        
    def sample(self, n=1):
        sigma = self._rho_to_sigma(self.rho)
        epsilon = torch.randn(size=(n,self.nb_dim)).to(self.device)
        return epsilon.mul(sigma).add(self.mu)
    
    def _rho_to_sigma(self, rho):
        sigma = torch.log(1 + torch.exp(rho))
        return sigma

    def _sigma_to_rho(self, sigma):
        rho = torch.log(torch.exp(sigma) - 1)
        return rho
    
    def requires_grad_(self, b):
        self.mu.requires_grad_(b)
        self.rho.requires_grad_(b)
    
    def log_prob(self, z):
        S = torch.diag(self.sigma)
        return torch.distributions.multivariate_normal.MultivariateNormal(self.mu, scale_tril=S).log_prob(z).unsqueeze(-1)

'''
class VariationalOptimizer():
    def __init__(self, learning_rate, patience, factor, device='cpu', min_lr=0.00001):
        self.device = device
        self.min_lr = min_lr
        self.learning_rate = learning_rate
        self.patience = patience
        self.factor = factor
            
    def run(self, q, logposterior, n_epoch=10000, n_ELBO_samples=1, plot=False):
        
        optimizer = torch.optim.Adam(q.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, factor=self.factor)
        
        if plot:
            liveloss = PlotLosses()
            
        for t in range(n_epoch):
            logs = {}
            optimizer.zero_grad()

            z = q.sample(n_ELBO_samples)
            LQ = q.log_prob(z)
            LP = logposterior(z)
            L = (LQ - LP).sum()/n_ELBO_samples

            L.backward()

            learning_rate = optimizer.param_groups[0]['lr']

            scheduler.step(L.detach().clone().cpu().numpy())
            logs['ELBO'] = L.detach().clone().cpu().numpy()
            logs['learning rate'] = learning_rate
            liveloss.update(logs)

            if plot and t % 10 == 0:
                liveloss.draw()

            optimizer.step()

            if learning_rate < 0.0001:
                return q
        return q

 '''   
