import torch
from torch import nn
import math
from tqdm import tqdm, trange



class MeanFieldVariationalDistribution(nn.Module):
    def __init__(self, nb_dim, mu=0.0, sigma=1.0, device='cpu'):
        super(MeanFieldVariationalDistribution, self).__init__()
        self.device = device
        self.nb_dim = nb_dim

        self.rho = nn.Parameter(torch.Tensor(nb_dim).to(self.device), requires_grad=True)

        if not torch.is_tensor(mu):
            mu = torch.tensor(mu)
            self.mu = nn.Parameter(torch.Tensor(nb_dim).to(self.device), requires_grad=True)
            nn.init.constant_(self.mu, mu)
        else:
            self.mu = nn.Parameter(mu.to(self.device), requires_grad=True)

        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma)
            rho = torch.log(torch.exp(sigma) - 1)
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

    def log_prob(self, x):
        S = self.sigma
        mu = self.mu
        dim=self.nb_dim
        n_x=x.shape[0]
        H=S.view(dim,1,1).inverse().view(1,1,dim)
        d=((x-mu.view(1,dim))**2).view(n_x,dim)
        const=0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
        return -0.5*(H*d).sum(2).squeeze()-const

    # def log_prob(self, z):
    #     S = torch.diag(self.sigma)
    #     return torch.distributions.multivariate_normal.MultivariateNormal(self.mu, scale_tril=S).log_prob(z).unsqueeze(-1)


class MeanFieldVariationInference():
    def __init__(self, objective_fn, max_iter, n_ELBO_samples, learning_rate, min_lr, patience, lr_decay, device, temp_dir):
        self.objective_fn = objective_fn
        self.max_iter = max_iter
        self.n_ELBO_samples = n_ELBO_samples
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device

        self.tempdir_name = temp_dir


        self._best_score = float('inf')

    def _save_best_model(self, q, epoch, score,  ED, LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': q.state_dict(),
                'ELBO': score,
                'ED': ED,
                'LP': LP
            }, self.tempdir_name+'best.pt')
            self._best_score = score

    def _get_best_model(self, q):
        best = torch.load(self.tempdir_name+'best.pt')
        q.load_state_dict(best['state_dict'])
        return best['epoch'], [best['ELBO'], best['ED'], best['LP']]

    def run(self, q):
        q.rho.requires_grad = True
        q.mu.requires_grad = True
        optimizer = torch.optim.Adam(q.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,
                                                               factor=self.lr_decay)
        #self._best_q = theta.detach().clone().cpu().numpy()
        #self._best_score = np.inf
        self.score_elbo = []
        self.score_entropy = []
        self.score_logposterior = []
        self.score_lr = []

        with trange(self.max_iter) as tr:
            for t in tr:
                optimizer.zero_grad()

                theta = q.sample(self.n_ELBO_samples)
                LQ = q.log_prob(theta).squeeze().mean()
                LP = self.objective_fn(theta).mean()
                loss = LQ - LP

                loss.backward()

                lr = optimizer.param_groups[0]['lr']

                tr.set_postfix(ELBO=loss.item(), ED=-LQ.item(), lr=lr)

                #score.append(loss.detach().clone().cpu().numpy())
                scheduler.step(loss.detach().clone().cpu().numpy())
                optimizer.step()

                self._save_best_model(q,t, loss.detach().clone(), -LQ.detach().clone(), LP.detach().clone())

                if t%100== 0:
                    self.score_elbo.append(loss.detach().clone().cpu())
                    self.score_entropy.append(-LQ.detach().clone().cpu())
                    self.score_logposterior.append(LP.detach().clone().cpu())
                    self.score_lr.append(lr)

                if lr < self.min_lr:
                    break

        #best_theta, best_score = self._get_best_model()
        best_epoch, scores=self._get_best_model(q)
        return best_epoch, scores






