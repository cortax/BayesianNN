import torch
from torch import nn
import math
from tqdm import tqdm, trange

from Tools import NNE

class GeNNeVI():
    def __init__(self, objective_fn,
                 kNNE, n_samples_NNE, n_samples_LP,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device, temp_dir, save_best=True):
        self.objective_fn = objective_fn
        self.kNNE=kNNE
        self.n_samples_NNE=n_samples_NNE
        self.n_samples_LP=n_samples_LP
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device

        self.save_best=save_best
        self._best_score=float('inf')
    

        self.tempdir_name = temp_dir

    def ELBO(self,GeN):
        ED = NNE(GeN(self.n_samples_NNE),k=self.kNNE, device=self.device)
        LP = self.objective_fn(GeN(self.n_samples_LP)).mean()
        L = -ED - LP
        return L

    def _save_best_model(self, GeN, epoch, score,ED,LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': GeN.state_dict(),
                'ELBO': score
            }, self.tempdir_name+'/best.pt')
            self._best_score=score

    def _get_best_model(self, GeN):
        best= torch.load(self.tempdir_name+'/best.pt')
        GeN.load_state_dict(best['state_dict'])
        return best['epoch'], best['ELBO']

    def run(self, GeN, show_fn=None):
        optimizer = torch.optim.Adam(GeN.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,
                                                               factor=self.lr_decay)
        self.scores={'ELBO': [] ,
                     'Entropy':[],
                     'lr':[]    
        }
       

        with trange(self.max_iter) as tr:
            for t in tr:

                optimizer.zero_grad()

                ED = NNE(GeN(self.n_samples_NNE),k=self.kNNE, device=self.device)
                LP = self.objective_fn(GeN(self.n_samples_LP)).mean()
                L = -ED - LP
                L.backward()

                lr = optimizer.param_groups[0]['lr']

                scheduler.step(L.detach().clone().cpu().numpy())

                tr.set_postfix(ELBO=L.item(), ED=ED.item(), lr=lr)

                if t % 100 ==0:
                    self.scores['ELBO'].append(L.item())
                    self.scores['Entropy'].append(ED.item())
                    self.scores['lr'].append(lr)

                if self.save_best:
                    self._save_best_model(GeN, t, L.item(), ED.item(), LP.item())

                if lr < self.min_lr:
                    self._save_best_model(GeN, t, L.item(), ED.item(), LP.item())
                    break

                if t+1==self.max_iter:
                    self._save_best_model(GeN, t, L.item(), ED.item(), LP.item())

                optimizer.step()
                
        best_epoch, best_elbo =self._get_best_model(GeN)
        return best_epoch, best_elbo