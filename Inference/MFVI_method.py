import numpy as np
import math
import torch
from torch import nn
from torch import functional as F

from Prediction.metrics import get_logposterior, log_metrics, seeding

import argparse
import mlflow
import mlflow.pytorch


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


def main(get_data,get_model,sigma_noise,experiment_name,nb_split,ensemble_size, max_iter, learning_rate, min_lr,  n_samples_ED, n_samples_LP, patience, lr_decay, init_std, optimize, seed, device, verbose, show_metrics):
    seeding(seed)

    xpname = experiment_name + '/MFVI'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run():#run_name='eMFVI', experiment_id=expdata.experiment_id
        mlflow.set_tag('device', device)
        mlflow.set_tag('seed', seed)


        
        X_train, y_train, y_train_un, X_test, y_test_un, inverse_scaler_y = get_data(nb_split, device)
        
        mlflow.log_param('sigma noise', sigma_noise)
        mlflow.log_param('split', nb_split)
        
        param_count, mlp=get_model()
        mlflow.set_tag('dimensions', param_count)

        logtarget=get_logposterior(mlp,X_train,y_train,sigma_noise,device)

        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('learning_rate', learning_rate)

        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)

        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)

        mlflow.log_param('init_std', init_std)

        MFVI=MeanFieldVariationalMixtureDistribution(ensemble_size,param_count,init_std,device=device)
        

        optimizer = torch.optim.Adam(MFVI.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

        for t in range(max_iter):
            optimizer.zero_grad()

            ED=-MFVI.log_prob(MFVI(n_samples_ED)).mean()
            LP=logtarget(MFVI(n_samples_LP)).mean()
            L =-ED-LP
            L.backward()

            lr = optimizer.param_groups[0]['lr']
            scheduler.step(L.detach().clone().cpu().numpy())
            
            mlflow.log_metric("ELBO", float(L.detach().squeeze().clone().cpu().numpy()),t)
            mlflow.log_metric("-log posterior", float(-LP.detach().squeeze().clone().cpu().numpy()),t)
            mlflow.log_metric("differential entropy", float(ED.detach().clone().cpu().numpy()),t)
            mlflow.log_metric("learning rate", float(lr),t)
            mlflow.log_metric("epoch", t)

            
            if show_metrics:
                with torch.no_grad():
                    theta = MFVI(100)
                    log_metrics(theta, mlp, X_train, y_train_un, X_test, y_test_un, sigma_noise, inverse_scaler_y, t,device)

            if verbose:
                stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
                print(stats)

            if lr < min_lr:
                break

            optimizer.step()

        theta=MFVI(1000).detach()
        log_metrics(theta, mlp, X_train, y_train_un, X_test, y_test_un, sigma_noise, inverse_scaler_y, t,device)
        
        

        
parser = argparse.ArgumentParser()
parser.add_argument("--ensemble_size", type=int, default=1,
                    help="number of model to train in the ensemble")
parser.add_argument("--max_iter", type=int, default=100000,
                    help="maximum number of learning iterations")
parser.add_argument("--learning_rate", type=float, default=0.01,
                    help="initial learning rate of the optimizer")
parser.add_argument("--n_samples_ED", type=int, default=50,
                    help="number of samples for MC estimation of differential entropy")
parser.add_argument("--n_samples_LP", type=int, default=100,
                    help="number of samples for MC estimation of expected logposterior")   
parser.add_argument("--min_lr", type=float, default=0.0005,
                    help="minimum learning rate triggering the end of the optimization")
parser.add_argument("--patience", type=int, default=10,
                    help="scheduler patience")
parser.add_argument("--lr_decay", type=float, default=0.1,
                    help="scheduler multiplicative factor decreasing learning rate when patience reached")
parser.add_argument("--init_std", type=float, default=1.0,
                    help="parameter controling initialization of theta")
parser.add_argument("--optimize", type=int, default=0,
                    help="number of optimization iterations to initialize the state")
parser.add_argument("--expansion", type=int, default=0,
                    help="variational inference is done only on variance (0,1)")
parser.add_argument("--seed", type=int, default=None,
                    help="scheduler patience")
parser.add_argument("--device", type=str, default=None,
                    help="force device to be used")
parser.add_argument("--verbose", type=bool, default=False,
                    help="force device to be used")
parser.add_argument("--show_metrics", type=bool, default=False,
                    help="log metrics during training")    

