import torch
from torch import nn
import math

from tqdm import trange


from Tools import KL, JSD





class GeNPredVI():
    def __init__(self, loglikelihood, prior, projection, k_MC,
                 kNNE, n_samples_KL, n_samples_LL,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device, verbose, temp_dir, save_best=True):
        self.loglikelihood=loglikelihood
        self.prior=prior
        self.projection=projection
        self.k_MC=k_MC
        self.kNNE=kNNE
        self.n_samples_KL=n_samples_KL
        self.n_samples_LL=n_samples_LL
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device
        self.verbose = verbose

        self.save_best=save_best
        self._best_score=float('inf')


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

                theta_proj=self.projection(GeN(self.n_samples_KL),self.k_MC)

                theta_prior_proj=self.projection(self.prior(self.n_samples_KL),self.k_MC)

                #K=KL(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device)
                K=JSD(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device,p=1)

                LL = self.loglikelihood(GeN(self.n_samples_LL)).mean()
                L = K - LL
                L.backward()

                lr = optimizer.param_groups[0]['lr']

                scheduler.step(L.detach().clone().cpu())

                tr.set_postfix(ELBO=L.detach().cpu().float().numpy(), LogLike=LL.detach().cpu().float().numpy(), KL=K.detach().cpu().float().numpy(), lr=lr)

                if t % 100 ==0:
                    self.score_elbo.append(L.detach().clone().cpu())
                    self.score_KL.append(K.detach().clone().cpu())
                    self.score_LL.append(LL.detach().clone().cpu())
                    self.score_lr.append(lr)

                    if show_fn is not None:
                        #print('show')
                        show_fn(GeN,500)

                if self.save_best:
                    self._save_best_model(GeN, t,L.detach().clone(), K.detach().clone(), LL.detach().clone())

                if lr < self.min_lr:
                    self._save_best_model(GeN, t, L.detach().clone(), K.detach().clone(), LL.detach().clone())
                    break

                if t+1==self.max_iter:
                    self._save_best_model(GeN, t, L.detach().clone(), K.detach().clone(), LL.detach().clone())

                optimizer.step()

            
        best_epoch, scores =self._get_best_model(GeN)
        return best_epoch, scores