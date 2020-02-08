import numpy as np
import torch
from torch import nn
import math
import argparse
import mlflow
import mlflow.pytorch

from Prediction.metrics import get_logposterior,log_metrics,log_split_metrics



class HNet(nn.Module):
            def __init__(self, lat_dim, nb_neur, output_dim,  activation=nn.ReLU(), init_w=.1, init_b=.1, device='cpu'):
                super(HNet, self).__init__()
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
                epsilon = torch.randn(size=(n,self.lat_dim)).to(self.device)
                return self.hnet(epsilon)           

class HyNetEns(nn.Module):
    def __init__(self,nb_comp,lat_dim,layer_width, output_dim, activation, init_w,init_b,device='cpu'):
        super(HyNetEns, self).__init__()
        self.nb_comp=nb_comp
        self.output_dim=output_dim
        self.components= nn.ModuleList([HNet(lat_dim,layer_width,output_dim,activation,init_w,init_b,device) for i in range(nb_comp)]).to(device)   
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
    def KDE(x,y,prec):
        """
        KDE    

        Parameters:
            x (Tensor): Inputs, NbExemples X NbDimensions   
            y (Tensor):  Batched samples, NbBatch x NbSamples X NbDimensions
            prec (Float): scalar factor for bandwidth scaling


        Returns:
            (Tensor) KDE log estimate for x based on batched diagonal "Silverman's rule of thumb", NbExemples
            See Wand and Jones p.111 "Kernel Smoothing" 1995.  

        """

        dim=x.shape[-1]
        n_ed=x.shape[0]
        n_comp=y.shape[0]
        n_kde=y.shape[1]
        c_=(n_kde*(dim+2))/4
        c=torch.as_tensor(c_).pow(2/(dim+4)).to(device)  
        H=prec*(y.var(1)/c).clamp(torch.finfo().eps,float('inf'))

        d=((y.view(n_comp,n_kde,1,dim)-x.view(1,1,n_ed,dim))**2)
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





def main(get_data,get_model,sigma_noise,experiment_name,nb_split,ensemble_size,lat_dim,HN_layerwidth,init_w, kNNE, n_samples_NNE, n_samples_KDE, n_samples_ED, n_samples_LP, max_iter, learning_rate, min_lr, patience, lr_decay,  device=None, verbose=False,show_metrics=False,init_b=.001,HN_activation=nn.ReLU(),KDE_prec=1.):
    
    xpname = experiment_name+'/GeNVI'             
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)
    
    if kNNE==0:
            entropy_mthd='KDE'
    else:
            entropy_mthd=str(kNNE)+'NNE'
    
    with mlflow.start_run():
        mlflow.set_tag('device', device) 
        mlflow.set_tag('method', entropy_mthd)
        
        

        
        param_count, mlp=get_model()  
        
        
        mlflow.log_param('sigma noise', sigma_noise)
        mlflow.log_param('split nb', nb_split)
       
        
        mlflow.set_tag('dimensions', param_count)

         
        KDE=get_KDE(device)
        NNE=get_NNE(device)
        
        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('HyperNet_lat_dim', lat_dim)
        mlflow.log_param('HyperNet_layerwidth', HN_layerwidth)
       
         
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)
        
                
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)
        

        mlflow.log_param('n_samples_KDE', n_samples_KDE)
        mlflow.log_param('n_samples_ED', n_samples_ED)
        mlflow.log_param('n_samples_LP', n_samples_LP)
        
        
        splitting_metrics=[]
                
        for split in range(nb_split):
            with mlflow.start_run(run_name='split '+str(split), nested=True):
       
                X_train, y_train, y_train_un, X_test, y_test_un, inverse_scaler_y = get_data(split, device) 
                logtarget=get_logposterior(mlp,X_train,y_train,sigma_noise,device)

                Hyper_Nets=HyNetEns(ensemble_size, lat_dim, HN_layerwidth, param_count, HN_activation, init_w, init_b,device)

                optimizer = torch.optim.Adam(Hyper_Nets.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)
                
                              
                for t in range(max_iter):
                    optimizer.zero_grad()

                    if kNNE==0:
                        ED=-KDE(Hyper_Nets(n_samples_ED),Hyper_Nets.sample(n_samples_KDE),KDE_prec).mean()
                    else:
                        ED=NNE(Hyper_Nets(n_samples_NNE),kNNE)

                    LP=logtarget(Hyper_Nets(n_samples_LP)).mean()
                    L =-ED-LP
                    L.backward()


                    lr = optimizer.param_groups[0]['lr']

                    mlflow.log_metric("ELBO", float(L.detach().clone().cpu().numpy()),t)
                    mlflow.log_metric("-log posterior", float(-LP.detach().clone().cpu().numpy()),t)
                    mlflow.log_metric("differential entropy", float(ED.detach().clone().cpu().numpy()),t)
                    mlflow.log_metric("learning rate", float(lr),t)
                    mlflow.log_metric("epoch", t)


                    if show_metrics:
                        theta=Hyper_Nets(100).detach()
                        log_metrics(theta, mlp, X_train, y_train_un, X_test, y_test_un, sigma_noise, inverse_scaler_y, t,device)



                    scheduler.step(L.detach().clone().cpu().numpy())

                    if verbose:
                        stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
                        print(stats)

                    if lr < min_lr:
                        break


                    optimizer.step()


                theta=Hyper_Nets(1000).detach()
                splitting_metrics.append(log_metrics(theta, mlp, X_train, y_train_un, X_test, y_test_un, sigma_noise, inverse_scaler_y, t,device))
                mlflow.pytorch.log_model(Hyper_Nets,'models')

        log_split_metrics(splitting_metrics)


parser = argparse.ArgumentParser()
parser.add_argument("--ensemble_size", type=int, default=1,
                    help="number of hypernets to train in the ensemble")
parser.add_argument("--lat_dim", type=int, default=5,
                    help="number of latent dimensions of each hypernet")
parser.add_argument("--layerwidth", type=int, default=50,
                    help="layerwidth of each hypernet")
parser.add_argument("--init_w", type=float, default=0.2,
                    help="std for weight initialization of output layers")
#    parser.add_argument("--init_b", type=float, default=0.000001,
#                        help="std for bias initialization of output layers")  
parser.add_argument("--NNE", type=int, default=0,
                    help="k≥1 Nearest Neighbor Estimate, 0 is for KDE")
parser.add_argument("--n_samples_NNE", type=int, default=500,
                    help="number of samples for NNE")
parser.add_argument("--n_samples_KDE", type=int, default=1000,
                    help="number of samples for KDE")
parser.add_argument("--n_samples_ED", type=int, default=50,
                    help="number of samples for MC estimation of differential entropy")
parser.add_argument("--n_samples_LP", type=int, default=100,
                    help="number of samples for MC estimation of expected logposterior")
parser.add_argument("--max_iter", type=int, default=100000,
                    help="maximum number of learning iterations")
parser.add_argument("--learning_rate", type=float, default=0.08,
                    help="initial learning rate of the optimizer")
parser.add_argument("--min_lr", type=float, default=0.00000001,
                    help="minimum learning rate triggering the end of the optimization")
parser.add_argument("--patience", type=int, default=400,
                    help="scheduler patience")
parser.add_argument("--lr_decay", type=float, default=.5,
                    help="scheduler multiplicative factor decreasing learning rate when patience reached")
parser.add_argument("--device", type=str, default=None,
                    help="force device to be used")
parser.add_argument("--verbose", type=bool, default=False,
                    help="force device to be used")
parser.add_argument("--show_metrics", type=bool, default=False,
                    help="log metrics during training")    
"""
args = parser.parse_args()
 
    if args.device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device   
    
    print(args)
"""