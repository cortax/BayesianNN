import numpy as np
import torch
from torch import nn
import math
import argparse
import mlflow
import mlflow.pytorch




class GeNet(nn.Module):
            def __init__(self, lat_dim, nb_neur, output_dim,  activation=nn.ReLU(), init_w=.1, init_b=.1, device='cpu'):
                super(GeNet, self).__init__()
                self.lat_dim = lat_dim
                self.device=device
                self.output_dim=output_dim
                self.hnet=nn.Sequential(
                        nn.Linear(lat_dim,nb_neur),
                        activation,
                        nn.Linear(nb_neur,output_dim)
                        ).to(device)
                
                torch.nn.init.normal_(self.hnet[2].weight,mean=0., std=init_w)
                torch.nn.init.normal_(self.hnet[2].bias,mean=0., std=init_b)
    
            def forward(self, n=1):
                epsilon = torch.randn(size=(n,self.lat_dim), device=self.device)
                return self.hnet(epsilon)           

class GeNetEns(nn.Module):
    def __init__(self, nb_comp, lat_dim, layer_width, output_dim, activation, init_w, init_b, device):
        super(GeNetEns, self).__init__()
        self.nb_comp=nb_comp
        self.output_dim=output_dim
        self.components= nn.ModuleList([GeNet(lat_dim,layer_width,output_dim,activation,init_w,init_b,device) for i in range(nb_comp)]).to(device)

        self._best_compnents = None
        self._best_score = float('inf')

    def sample(self, n=1):
        return torch.stack([self.components[c](n) for c in range(self.nb_comp)])

    def _save_best_model(self, score,epoch,ED,LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'ELBO': score,
                'ED':ED,
                'LP':LP
            }, 'best.pt')
            self._best_score=score

    def _get_best_model(self):
        best= torch.load('best.pt')
        self.load_state_dict(best['state_dict'])
        return best['epoch'], best['ELBO'], best['ED'], best['LP']
    
    def forward(self, n=1):
        d = torch.distributions.multinomial.Multinomial(n, torch.ones(self.nb_comp))
        m = d.sample()
        return torch.cat([self.components[c](int(m[c])) for c in range(len(self.components))])
    
    
"""
use:

Hyper_Nets=HyNetEns(ensemble_size,lat_dim,HN_layerwidth, output_dim,activation,init_w,init_b).to(device)

Hyper_Nets(100)

Hyper_Nets.sample(100)

"""



### Entropy approximation

def get_KDE(device):
    def KDE(x, x_kde):
        """
        KDE    

        Parameters:
            x (Tensor): Inputs, NbExemples X NbDimensions   
            x_kde (Tensor):  Batched samples, NbBatch x NbSamples X NbDimensions


        Returns:
            (Tensor) KDE log estimate for x based on batched diagonal "Silverman's rule of thumb", NbExemples
            See Wand and Jones p.111 "Kernel Smoothing" 1995.  

        """

        dim=x.shape[-1]
        n_ed=x.shape[0]
        n_comp=x_kde.shape[0]
        n_kde=x_kde.shape[1]
        c_=(n_kde*(dim+2))/4
        c=torch.as_tensor(c_).pow(2/(dim+4)).to(device)  
        H=(x_kde.var(1) / c).clamp(torch.finfo().eps, float('inf'))

        d=((x_kde.view(n_comp, n_kde, 1, dim) - x.view(1, 1, n_ed, dim)) ** 2)
        H_=H.view(n_comp,dim,1,1).inverse().view(n_comp,1,1,dim)
        const=0.5*H.log().sum(1)+0.5*dim*torch.tensor(2*math.pi).log()
        const=const.view(n_comp,1,1)
        ln=-0.5*(H_*d).sum(3)-const
        N=torch.as_tensor(float(n_comp*n_kde),device=device)
        return (ln.logsumexp(0).logsumexp(0)-torch.log(N)).unsqueeze(-1)
    return KDE

def get_NNE(device):
    def NNE(theta,k=1):
        """
        Parameters:
            theta (Tensor): Samples, NbExemples X NbDimensions   
            k (Int): ordinal number

        Returns:
            (Float) k-Nearest Neighbour Estimation of the entropy of theta  

        """
        nb_samples=theta.shape[0]
        dim=theta.shape[1]
        D=torch.cdist(theta,theta)
        a = torch.topk(D, k=k+1, dim=0, largest=False, sorted=True)[0][k].clamp(torch.finfo().eps,float('inf')).to(device)
        d=torch.as_tensor(float(dim),device=device)
        K=torch.as_tensor(float(k),device=device)
        N=torch.as_tensor(float(nb_samples),device=device)
        pi=torch.as_tensor(math.pi,device=device)
        lcd = d/2.*pi.log() - torch.lgamma(1. + d/2.0)
        return torch.log(N) - torch.digamma(K) + lcd + d/nb_samples*torch.sum(torch.log(a))
    return NNE

def get_entropy(kNNE,n_samples_ED,n_samples_KDE,n_samples_NNE,device):
    if kNNE == 0:
        def entropy(GeN):
            KDE=get_KDE(device)
            return -KDE(GeN(n_samples_ED), GeN.sample(n_samples_KDE)).mean()
        return entropy
    else:
        def entropy(GeN):
            NNE=get_NNE(device)
            return NNE(GeN(n_samples_NNE), kNNE)
        return entropy



def GeNVI(objective_fn,
          GeN, kNNE, n_samples_NNE, n_samples_KDE, n_samples_ED, n_samples_LP,
          max_iter, learning_rate, min_lr, patience, lr_decay,
          device=None, verbose=False):

    """
    GeNVI method

        objective_fn : S X DIM -> S

        GeN (nn.Module) with methods:
            forward(N): sample of size  N X DIM
            sample(N): ensemble sample of size   ENS X N X DIM
            # TODO: add description
            _save_best_model
            _get_best_model

    """

    entropy = get_entropy(kNNE, n_samples_ED, n_samples_KDE, n_samples_NNE, device)


    optimizer = torch.optim.Adam(GeN.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

    for t in range(max_iter):
        optimizer.zero_grad()

        ED = entropy(GeN)
        LP = objective_fn(GeN(n_samples_LP)).mean()
        L = -ED - LP
        L.backward()

        lr = optimizer.param_groups[0]['lr']


        scheduler.step(L.detach().clone().cpu().numpy())

        if verbose:
            stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
            print(stats)

        GeN._save_best_model(L, t, ED, LP)

        if lr < min_lr:
            break

        optimizer.step()

    return