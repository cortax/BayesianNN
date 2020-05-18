import torch
from torch import nn
import math

from tqdm import trange


from Tools import KL, batchKL


class FuNNeVI():
    def __init__(self, loglikelihood, prior, projection, n_samples_FU, ratio_ood, p,
                 kNNE, n_samples_KL, n_samples_LL,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device,  temp_dir, save_best=True):
        self.loglikelihood=loglikelihood
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

        self.save_best=save_best
        self._best_score=float('inf')
        
        self.p=p


        self.tempdir_name = temp_dir

    def _save_best_model(self, GeN, epoch, score,ED,LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': GeN.state_dict(),
                'ELBO': score,
                'ED':ED,
                'LP':LP
            }, self.tempdir_name+'/best.pt')
            self._best_score=score

    def ELBO(self,GeN,m_MCL=100,n_LL=1000):
        #compute ELBO of GeN accurately
        
        theta=GeN(self.n_samples_KL)
        theta_prior=self.prior(self.n_samples_KL)
        theta_proj=torch.Tensor(m_MCL, self.n_samples_KL, self.n_samples_FU).to(self.device)
        theta_prior_proj=torch.Tensor(m_MCL, self.n_samples_KL, self.n_samples_FU).to(self.device)
        for i in range(m_MCL):
            theta_proj[i], theta_prior_proj[i] = self.projection(theta, theta_prior, self.n_samples_FU, self.ratio_ood)
        
        K=batchKL(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device,p=self.p)
        LL = self.loglikelihood(GeN(n_LL)).mean()
        L = K - LL
        return L
    
    def _KL(self,GeN):
        
        theta=GeN(self.n_samples_KL)
        theta_prior=self.prior(self.n_samples_KL)

        theta_proj, theta_prior_proj = self.projection(theta, theta_prior, self.n_samples_FU, self.ratio_ood)

        K=KL(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device,p=self.p)
        return K

        
             
    def _get_best_model(self, GeN):
        best= torch.load(self.tempdir_name+'/best.pt')
        GeN.load_state_dict(best['state_dict'])
        return best['epoch'], [best['ELBO'], best['ED'], best['LP']]

    def run(self, GeN, show_fn=None):
        optimizer = torch.optim.Adam(GeN.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, factor=self.lr_decay)

        self.score_elbo = []
        self.score_KL = []
        self.score_LL = []
        self.score_lr = []
        with trange(self.max_iter) as tr:
            for t in tr:
                optimizer.zero_grad()
                """
                theta=GeN(self.n_samples_KL)
                theta_prior=self.prior(self.n_samples_KL)

                theta_proj, theta_prior_proj=self.projection(theta, theta_prior, self.n_samples_FU, self.ratio_ood)
                K=KL(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device)

                LL = self.loglikelihood(GeN(self.n_samples_LL)).mean()
                L = K - LL
                """
                #TODO explore hyperparameter lambda*K
                K = self._KL(GeN)
                LL = self.loglikelihood(GeN(self.n_samples_LL)).mean()
                L=K-LL
                L.backward()

                lr = optimizer.param_groups[0]['lr']

                scheduler.step(L.item())

                tr.set_postfix(ELBO=L.item(), LogLike=LL.item(), KL=K.item(), lr=lr)

                if t % 100 ==0:
                    self.score_elbo.append(L.detach().clone().cpu())
                    self.score_KL.append(K.detach().clone().cpu())
                    self.score_LL.append(LL.detach().clone().cpu())
                    self.score_lr.append(lr)

                if self.save_best:
                    self._save_best_model(GeN, t,L.item(), K.item(), LL.item())

                if lr < self.min_lr:
                    self._save_best_model(GeN, t, L.item(), K.item(), LL.item())
                    break

                if t+1==self.max_iter:
                    self._save_best_model(GeN, t, L.item(), K.item(), LL.item())

                optimizer.step()

            
        best_epoch, scores =self._get_best_model(GeN)
        return best_epoch, scores