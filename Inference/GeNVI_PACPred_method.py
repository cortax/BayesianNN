import torch
from torch import nn
import math




class GeNet(nn.Module):
            def __init__(self, lat_dim, nb_neur, output_dim,  activation, init_w, init_b, device):
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
        self.device = device
        self.nb_comp=nb_comp
        self.output_dim=output_dim
        self.components= nn.ModuleList([GeNet(lat_dim,layer_width,output_dim,activation,init_w,init_b,device) for i in range(nb_comp)]).to(device)

        self._best_compnents = None
        self._best_score = float('inf')

    def sample(self, n=1):
        return torch.stack([self.components[c](n) for c in range(self.nb_comp)])


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


def NNE(theta,device,k_MC,k=1):
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
    d=torch.as_tensor(float(dim), device=device)
    K=torch.as_tensor(float(k), device=device)
    K_MC=torch.as_tensor(float(k_MC), device=device)
    N=torch.as_tensor(float(nb_samples), device=device)
    pi=torch.as_tensor(math.pi, device=device)
    lcd = d/2.*pi.log() - torch.lgamma(1. + d/2.0)-d/2*K_MC.log()
    return torch.log(N) - torch.digamma(K) + lcd + d/nb_samples*torch.sum(torch.log(a))



def log1m_exp(a):
    if (a >= 0):
        return float('nan');
    elif a > -0.693147:
        return torch.log(-torch.expm1(a)) #0.693147 is approximatelly equal to log(2)
    else:
        return torch.log1p(-torch.exp(a))

def lograt1mexp(x,y):
    ''' 
   
    '''
    M=x#torch.max(x,y)
    m=y#torch.min(x,y)
    lor=y-x+log1m_exp(-x)-log1m_exp(-y)
    return lor

class GeNPACPred():
    def __init__(self, loss,logprior, n_data_samples, C, R, projection,k_MC,
                 kNNE, n_samples_NNE, n_samples_KDE, n_samples_ED, n_samples_LP,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device, verbose, temp_dir, save_best=True):
        self.loss = loss
        self.logprior=logprior
        self.n_data_samples=n_data_samples
        self.C=torch.tensor(C,device=device)
        self.R=torch.tensor(R,device=device)
        self.projection=projection
        self.k_MC=k_MC

        self.kNNE=kNNE
        self.n_samples_NNE=n_samples_NNE
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

    
    def run(self, GeN, show_fn=None):
        _C=torch.log(torch.exp(self.C)-1).clone().to(self.device).detach().requires_grad_(True)
        optimizer = torch.optim.Adam(list(GeN.parameters()), lr=self.learning_rate)
        optimizer_temp=torch.optim.Adam([_C],lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,
                                                               factor=self.lr_decay)
        scheduler_temp = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_temp, patience=self.patience,
                                                               factor=self.lr_decay)


        self.score_elbo = []
        self.score_entropy = []
        self.score_temp = []
        self.score_lr = []
        
        for t in range(self.max_iter):
            optimizer.zero_grad()
            optimizer_temp.zero_grad()
            
            C = torch.log(torch.exp(_C) + 1.)
            delta = torch.tensor(0.05, device=self.device)

            theta_proj=self.projection(GeN(self.n_samples_NNE),self.k_MC)
            ED=NNE(theta_proj,self.device,self.k_MC,k=1)
            LP= self.logprior(theta_proj).mean()
            kl = -ED - LP

            loss = self.loss(GeN(self.n_samples_LP),self.R).mean()

            L = C * loss + (1 / self.n_data_samples) *  (kl + math.log(2 * math.sqrt(self.n_data_samples) / delta))

            B=(1-torch.exp(-L))/(1-torch.exp(-C))
            
            L.backward(retain_graph=True)
            B.backward()
            
            
            lr = optimizer.param_groups[0]['lr']
            lr_temp = optimizer_temp.param_groups[0]['lr']

            scheduler.step(L.detach().clone().cpu().numpy())
            scheduler_temp.step(B.detach().clone().cpu().numpy())
            
            if t % 100 ==0:
                self.score_elbo.append(loss.detach().clone().cpu())
                self.score_entropy.append(kl.detach().clone().cpu())
                self.score_temp.append(C.detach().clone().cpu())
                self.score_lr.append(lr)

                if show_fn is not None:
                    show_fn(GeN,500)

            if self.save_best:
                self._save_best_model(GeN, t,B.detach().clone(), loss.detach().clone(), kl.detach().clone(), C.detach().clone())

            if lr_temp < self.min_lr:
                self._save_best_model(GeN, t, B.detach().clone(), loss.detach().clone(), kl.detach().clone(), C.detach().clone())
                break

            if t+1==self.max_iter:
                self._save_best_model(GeN, t, B.detach().clone(), loss.detach().clone(), kl.detach().clone(), C.detach().clone())
            
            
            
         
            
            if self.verbose:
                stats = 'Epoch [{}/{}], Bound: {}, Entropy: {}, Temp: {}, KL: {}, Loss: {}, Learning Rate: {}'.format(t, self.max_iter, B, ED, C, kl, loss,lr)
                print(stats)
            
            optimizer.step()
            optimizer_temp.step()


        best_epoch, scores =self._get_best_model(GeN)
        return best_epoch, scores