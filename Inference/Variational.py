import torch
import numpy as np
from torch import nn
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


class MeanFieldVariationInference():
    def __init__(self, objective_fn, max_iter, n_ELBO_samples, learning_rate, min_lr, patience, lr_decay, device, verbose):
        self.objective_fn = objective_fn
        self.max_iter = max_iter
        self.n_ELBO_samples = n_ELBO_samples
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device
        self.verbose = verbose

        self._best_theta = None
        self._best_score = None
            
    def _save_best_model(self, score, theta):
        return
        if score < self._best_score:
            self._best_theta = theta
            self._best_score = score

    def _get_best_model(self):
        return
        return self._best_theta, self._best_score

    def run(self, q):
        q.rho.requires_grad = True
        q.mu.requires_grad = True
        optimizer = torch.optim.Adam(q.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,
                                                               factor=self.lr_decay)
        #self._best_q = theta.detach().clone().cpu().numpy()
        #self._best_score = np.inf
        score = []
        for t in range(self.max_iter - 1):
            optimizer.zero_grad()

            theta = q.sample(self.n_ELBO_samples)
            LQ = q.log_prob(theta).squeeze()
            LP = self.objective_fn(theta)
            loss = (LQ - LP).mean()

            loss.backward()

            lr = optimizer.param_groups[0]['lr']

            if self.verbose:
                stats = 'Epoch [{}/{}], Loss: {}, Learning Rate: {}'.format(t, self.max_iter, loss, lr)
                print(stats)

            score.append(loss.detach().clone().cpu().numpy())
            scheduler.step(loss.detach().clone().cpu().numpy())
            optimizer.step()

            #self._save_best_model(loss.detach().clone().cpu().numpy(), theta.detach().clone().cpu().numpy())

            if lr < self.min_lr:
                break

        #best_theta, best_score = self._get_best_model()
        return q






