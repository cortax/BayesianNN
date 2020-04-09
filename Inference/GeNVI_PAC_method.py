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






def KDE(x, x_kde,device):
    """
    KDE

    Parameters:
        x (Tensor): Inputs, NbSamples X NbDimensions
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
    N=torch.as_tensor(float(n_comp*n_kde), device=device)
    return (ln.logsumexp(0).logsumexp(0)-torch.log(N)).unsqueeze(-1)

def NNE(theta,device,k=1):
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
    N=torch.as_tensor(float(nb_samples), device=device)
    pi=torch.as_tensor(math.pi, device=device)
    lcd = d/2.*pi.log() - torch.lgamma(1. + d/2.0)
    ret=torch.log(N) - torch.digamma(K) + lcd + d/nb_samples*torch.sum(torch.log(a))
    return ret


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

class GeNPAC():
    def __init__(self, loss,logprior, n_data_samples, C, R,
                 kNNE, n_samples_NNE, n_samples_KDE, n_samples_ED, n_samples_LP,
                 max_iter, learning_rate, min_lr, patience, lr_decay,
                 device, verbose, temp_dir, save_best=True):
        self.loss = loss
        self.logprior=logprior
        self.n_data_samples=n_data_samples
        self.C=torch.tensor(C,device=device)
        self.R=torch.tensor(R,device=device)
        
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



    def _entropy(self, GeN):
        if self.kNNE == 0:
            ED = -KDE(GeN(self.n_samples_ED), GeN.sample(self.n_samples_KDE),self.device).mean()
        else:
            ED = NNE(GeN(self.n_samples_NNE),k=self.kNNE, device=self.device)
        return ED



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

            ED = self._entropy(GeN)
            loss = self.loss(GeN(self.n_samples_LP),self.R).mean()
            kl = -ED - self.logprior(GeN(self.n_samples_ED)).mean()

            # log(e^0 + e^-x) = max(0,-x) + log(e^0-max(-x, 0) + e^(-x - max(-x,0)))
            # log(1 - e^-x)=log(e^0 - e^-x) = log(e^min(-x,0)(e^ -min(-x,0) - e^ -x-min(-x,0))

            n = C * loss + (1 / self.n_data_samples) *  (kl + math.log(2 * math.sqrt(self.n_data_samples) / delta))
            #d = torch.log(1-torch.exp(-C))

            L =n
            B=(1-torch.exp(-n))/(1-torch.exp(-C)) #torch.log1p(-torch.exp(-n))-torch.log1p(-torch.exp(-C))
            
            L.backward(retain_graph=True)
            B.backward()
            
            
            lr = optimizer.param_groups[0]['lr']
            lr_temp = optimizer_temp.param_groups[0]['lr']

            scheduler.step(L.detach().clone().cpu().numpy())
            scheduler_temp.step(B.detach().clone().cpu().numpy())
            
            if t % 100 ==0:
                self.score_elbo.append(L.detach().clone().cpu())
                self.score_entropy.append(ED.detach().clone().cpu())
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