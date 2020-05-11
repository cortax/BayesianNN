import torch
from torch import nn
import math

from tqdm import trange

from Tools import KL


class GeNPACPred():
    def __init__(self, loss, prior, delta, n_data_samples, C, projection,k_MC,
                 kNNE, n_samples_KL, n_samples_L,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device,  temp_dir, save_best=True):
        self.loss = loss
        self.prior=prior
        self.n_data_samples=n_data_samples
        self.C=torch.tensor(C,device=device)
        self.delta = torch.tensor(delta, device=device)

        self.projection=projection
        self.k_MC=k_MC

        self.kNNE=kNNE
        self.n_samples_KL=n_samples_KL
        self.n_samples_L=n_samples_L
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device

        self.save_best=save_best
        self._best_score=float('inf')
        

        self.tempdir_name = temp_dir




    def _save_best_model(self, GeN, epoch, score, loss, kl, C):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': GeN.state_dict(),
                'Bound': score,
                'Loss':loss,
                'KL':kl,
                'Temp':C
            }, self.tempdir_name+'/best.pt')
            self._best_score=score

    def _get_best_model(self, GeN):
        best= torch.load(self.tempdir_name+'/best.pt')
        GeN.load_state_dict(best['state_dict'])
        return best['epoch'], [best['Bound'], best['Loss'], best['KL'],best['Temp']]

    
    def run(self, GeN, show_func):
        _C=torch.log(torch.exp(self.C)-1).clone().to(self.device).detach().requires_grad_(True)
        optimizer = torch.optim.Adam(list(GeN.parameters()), lr=self.learning_rate)
        optimizer_temp=torch.optim.Adam([_C],lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,
                                                               factor=self.lr_decay)
        #scheduler_temp = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_temp,patience=self.patience,
                                                               #factor=self.lr_decay)

        self.score_B = []
        self.score_KL = []
        self.score_Loss = []
        self.score_lr = []

        self.score_temp = []
        
        with trange(self.max_iter) as tr:
            for t in tr:
                optimizer.zero_grad()
                optimizer_temp.zero_grad()
                
                C = torch.log(torch.exp(_C) + 1.)
                

                theta_proj=self.projection(GeN(self.n_samples_KL),self.k_MC)

                theta_prior_proj=self.projection(self.prior(self.n_samples_KL),self.k_MC)

                kl=KL(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device)
                #JSD(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device,p=1)

                loss = self.loss(GeN(self.n_samples_L)).mean()

                L = C * loss + (1 / self.n_data_samples) *  (kl + math.log(2 * math.sqrt(self.n_data_samples) / self.delta))

                L.backward(retain_graph=True)
                
                optimizer.step()
                scheduler.step(L.detach().clone().cpu().numpy())

                
                B=(1-torch.exp(-L))/(1-torch.exp(-C))
                B.backward()
                
                optimizer_temp.step()
                
                #scheduler_temp.step(B.detach().clone().cpu().numpy())

                lr = optimizer.param_groups[0]['lr']
                lr_temp = optimizer_temp.param_groups[0]['lr']
                

                tr.set_postfix(ELBO=L.item(), B=B.item(), loss=loss.item(), 
                               KL=kl.item(),C=C.item(), lr=lr, lr_temp=lr_temp)


                
                

                
            
                if t % 100 ==0:
                    self.score_B.append(B.detach().clone().cpu())
                    self.score_Loss.append(loss.detach().clone().cpu())
                    self.score_KL.append(kl.detach().clone().cpu())
                    self.score_temp.append(C.detach().clone().cpu())
                    self.score_lr.append(lr)
                    show_func(GeN)

                if self.save_best:
                    self._save_best_model(GeN, t,B.detach().clone(), loss.detach().clone(), kl.detach().clone(), C.detach().clone())

                if lr < self.min_lr:
                    self._save_best_model(GeN, t, B.detach().clone(), loss.detach().clone(), kl.detach().clone(), C.detach().clone())
                    break

                if t+1==self.max_iter:
                    self._save_best_model(GeN, t, B.detach().clone(), loss.detach().clone(), kl.detach().clone(), C.detach().clone())


        best_epoch, scores =self._get_best_model(GeN)
        return best_epoch, scores