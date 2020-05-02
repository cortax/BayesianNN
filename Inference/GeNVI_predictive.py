import torch
from torch import nn
import math

from Tools import KDE, NNE

from Models import GeNet, GeNetEns



class GeNPredVI():
    def __init__(self, loglikelihood, logprior, projection, k_MC,
                 kNNE, n_samples_KDE, n_samples_ED, n_samples_LP,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device, verbose, temp_dir, save_best=True):
        self.loglikelihood=loglikelihood
        self.logprior=logprior
        self.projection=projection
        self.k_MC=k_MC
        self.kNNE=kNNE
        self.n_samples_KDE=n_samples_KDE
        self.n_samples_ED=n_samples_ED
        self.n_samples_LP=n_samples_LP
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,
                                                               factor=self.lr_decay)

        self.score_elbo = []
        self.score_entropy = []
        self.score_logposterior = []
        self.score_lr = []

        for t in range(self.max_iter):
            optimizer.zero_grad()
            
            theta_proj=self.projection(GeN(self.n_samples_ED),self.k_MC)
            if self.kNNE==0:
                theta_proj2=self.projection(GeN(self.n_samples_KDE),self.k_MC)
                ED=-KDE(theta_proj2,theta_proj.unsqueeze(0),self.device).mean()
            else:
                ED=NNE(theta_proj, k=self.kNNE, k_MC=self.k_MC, device=self.device)
            
            LP= self.logprior(theta_proj).mean()
            
            
            LL = self.loglikelihood(GeN(self.n_samples_LP)).mean()
            L = -ED - LP - LL
            L.backward()

            lr = optimizer.param_groups[0]['lr']

            scheduler.step(L.detach().clone().cpu().numpy())

            if self.verbose:
                stats = 'Epoch [{}/{}], Loss: {}, Entropy {}, Learning Rate: {}'.format(t, self.max_iter, L, ED, lr)
                print(stats)

            if t % 100 ==0:
                self.score_elbo.append(L.detach().clone().cpu())
                self.score_entropy.append(ED.detach().clone().cpu())
                self.score_logposterior.append(LP.detach().clone().cpu())
                self.score_lr.append(lr)

                if show_fn is not None:
                    #print('show')
                    show_fn(GeN,500)

            if self.save_best:
                self._save_best_model(GeN, t,L.detach().clone(), ED.detach().clone(), LP.detach().clone())

            if lr < self.min_lr:
                self._save_best_model(GeN, t, L.detach().clone(), ED.detach().clone(), LP.detach().clone())
                break

            if t+1==self.max_iter:
                self._save_best_model(GeN, t, L.detach().clone(), ED.detach().clone(), LP.detach().clone())

            optimizer.step()

        best_epoch, scores =self._get_best_model(GeN)
        return best_epoch, scores