import torch
from torch import nn
import math

from tqdm import trange


from Tools import KL, batchKL

#regularizer for the KL
rho=1.

class FuNNeVI():
    def __init__(self, loglikelihood, batch, size_data, prior, projection, n_samples_FU, ratio_ood, p,
                 kNNE, n_samples_KL, n_samples_LL,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device, show=None):
        self.loglikelihood=loglikelihood
        self.batch=batch
        self.size_data=size_data
        self.prior=prior
        self.projection=projection
        self.n_samples_FU=n_samples_FU
        self.ratio_ood=ratio_ood
        self.kNNE=kNNE
        self.n_samples_KL=n_samples_KL
        self.n_samples_LL=n_samples_LL
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device
        
        self.show=show
    
        
        self.p=p



    def ELBO(self,GeN,m_MCL=100,n_LL=200):
        #compute ELBO of GeN accurately
        device=self.device
        theta=GeN(self.n_samples_KL)
        theta_prior=self.prior(self.n_samples_KL)
        theta_proj=torch.Tensor(m_MCL, self.n_samples_KL, self.n_samples_FU).to(device)
        theta_prior_proj=torch.Tensor(m_MCL, self.n_samples_KL, self.n_samples_FU).to(device)
        for i in range(m_MCL):
            t, t_p= self.projection(theta, theta_prior, self.n_samples_FU, self.ratio_ood)
            theta_proj[i], theta_prior_proj[i]= t.detach().cpu(), t_p.detach().cpu()
        
        K=batchKL(theta_proj.detach().cpu(), theta_prior_proj.detach().cpu(),k=self.kNNE,device=device,p=self.p)
        LL = self.loglikelihood(GeN(n_LL).detach()).mean().to(device)
        L = K - LL
        return L
    
    def _KL(self,GeN):
        
        theta=GeN(self.n_samples_KL) #variationnel
        theta_prior=self.prior(self.n_samples_KL) #prior

        theta_proj, theta_prior_proj = self.projection(theta, theta_prior, self.n_samples_FU, self.ratio_ood)

        K=KL(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device,p=self.p)
        return K

        


    def run(self, GeN, show_fn=None):
        one_epoch=int(self.size_data/self.batch)
        
        optimizer = torch.optim.Adam(GeN.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, factor=self.lr_decay)

        self.scores={'ELBO': [] ,
                     'KL':[],
                     'LL':[],
                     'lr':[]    
        }
        
        
        
        with trange(self.max_iter) as tr:
            for t in tr:
                
                optimizer.zero_grad()
    
                K = self._KL(GeN) #KL(Q_var,Prior)
                LL = self.loglikelihood(GeN(self.n_samples_LL), self.batch).mean()
                L=(self.batch/self.size_data)*rho*K-LL
                L.backward()

                lr = optimizer.param_groups[0]['lr']
                
                
                tr.set_postfix(ELBO=L.item(), LogLike=LL.item(), KL=K.item(), lr=lr)

                optimizer.step()

                                
                if t % 100 ==0:
                    self.scores['ELBO'].append(L.item())
                    self.scores['KL'].append(K.item())
                    self.scores['LL'].append(LL.item())
                    self.scores['lr'].append(lr)
                    if self.show is not None:
                        self.show(GeN)

      
                if lr < self.min_lr:
                      break

                
                scheduler.step(L.item())


            return self.ELBO(GeN)