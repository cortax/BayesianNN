import torch
from torch import nn
import math
from tqdm import tqdm, trange

from Tools import KL

class GeNNeVI():
    def __init__(self, objective_fn, batch, size_data, prior,
                 kNNE, n_samples_KL, n_samples_LL,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device,  save_best=True):
        self.objective_fn = objective_fn
        self.batch=batch
        self.size_data=size_data
        self.prior=prior

        self.kNNE=kNNE
        self.n_samples_KL=n_samples_KL
        self.n_samples_LL=n_samples_LL
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device

        self._best_score=float('inf')
    


    def ELBO(self,GeN):
        theta=GeN(self.n_samples_KL) #variationnel
        theta_prior=self.prior(self.n_samples_KL) #prior

        K=KL(theta, theta_prior, k=self.kNNE, device=self.device,p=2) 
        LL = self.objective_fn(GeN(self.n_samples_LL)).mean()
        L = K - LL
        return L
    
    def _KL(self,GeN):
        
        theta=GeN(self.n_samples_KL) #variationnel
        theta_prior=self.prior(self.n_samples_KL) #prior

        K=KL(theta, theta_prior, k=self.kNNE, device=self.device,p=2)
        return (self.batch/self.size_data)*K


    def run(self, GeN, show_fn=None):
        one_epoch=int(self.size_data/self.batch)

        optimizer = torch.optim.Adam(GeN.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,
                                                               factor=self.lr_decay)
        self.scores={'ELBO': [] ,
                     'KL':[],
                     'LL':[],
                     'lr':[]    
        }
       

        with trange(self.max_iter) as tr:
            for t in tr:

                optimizer.zero_grad()

                K = self._KL(GeN) #KL(Q_var,Prior)
                LL = self.objective_fn(GeN(self.n_samples_LL), self.batch).mean()
                L = K - LL
                L.backward()

                lr = optimizer.param_groups[0]['lr']

                scheduler.step(L.detach().clone().cpu().numpy())

                tr.set_postfix(ELBO=L.item(),LL=LL.item(), KL=K.item(), lr=lr)

                if t % 100 ==0:
                    self.scores['ELBO'].append(L.item())
                    self.scores['KL'].append(K.item())
                    self.scores['LL'].append(LL.item())
                    self.scores['lr'].append(lr)

                if lr < self.min_lr:
                    break

                optimizer.step()
                
        return self.ELBO(GeN)