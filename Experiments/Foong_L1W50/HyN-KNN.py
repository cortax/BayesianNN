import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from Tools.NNtools import *
import tempfile
import mlflow
import Experiments.Foong_L1W50.setup as exp
import argparse
import pandas as pd
import math


class HNet(nn.Module):
            def __init__(self, lat_dim, nb_neur, output_dim,  activation=nn.ReLU(), init_w=.4, init_b=0.001):
                super(HNet, self).__init__()
                self.lat_dim = lat_dim
                self.output_dim=output_dim
                self.hnet=nn.Sequential(
                        nn.Linear(lat_dim,nb_neur),
                        activation,
                        nn.Linear(nb_neur,output_dim)
                        ).to(device)
                
                torch.nn.init.normal_(self.hnet[2].weight,mean=0., std=init_w)
                torch.nn.init.normal_(self.hnet[2].bias,mean=0., std=init_b)

    
            def forward(self, n=1):
                epsilon = torch.randn(size=(n,self.lat_dim)).to(device)
                return self.hnet(epsilon)           

class HyNetEns(nn.Module):
    def __init__(self,nb_comp,lat_dim, output_dim, activation, init_w,init_b):
        super(HyNetEns, self).__init__()
        self.nb_comp=nb_comp
        self.output_dim=output_dim
        self.components= nn.ModuleList([HNet(lat_dim,output_dim,output_dim,activation,init_w,init_b) for i in range(nb_comp)]).to(device)   

    def KNNE(self,nb_samples,k=1):  
        theta=self.forward(nb_samples)
        D=torch.cdist(theta,theta)
        a = torch.topk(D, k=k+1, dim=0, largest=False, sorted=True)[0][k].clamp(torch.finfo().eps,float('inf'))
        d=torch.as_tensor(float(self.output_dim)).to(device)
        K=torch.as_tensor(float(k)).to(device)
        N=torch.as_tensor(float(nb_samples)).to(device)
        lcd = (d/2.0*torch.log(torch.as_tensor(math.pi))) - torch.lgamma(1. + d/2.0)
        return torch.digamma(N) - torch.digamma(K) + lcd + d/nb_samples*torch.sum(torch.log(a))
    
 
    def sample(self, n=1):
        return torch.stack([self.components[c](n) for c in range(self.nb_comp)])

    
    def forward(self, n=1):
        return torch.cat([self.components[c](n).squeeze(0) for c in range(self.nb_comp)],dim=0)
    
        
def main(ensemble_size=1,lat_dim=5,init_w=.2,init_b=.001,n_samples_KNN=1000,n_samples_ED=20, n_samples_LP=20, max_iter=10000, learning_rate=0.001, min_lr=0.000005, patience=100, lr_decay=0.9,  device='cuda:1', verbose=True):
    
    activation=nn.ReLU()
    
    xpname = exp.experiment_name + 'HyNet-KNN'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(run_name='HyNet-KNN', experiment_id=expdata.experiment_id):
        mlflow.set_tag('device', device) 
        logposterior = exp.get_logposterior_parallel_fn(device)#new!
        model = exp.get_parallel_model(device) #new!
        x_train, y_train = exp.get_training_data(device)
        x_validation, y_validation = exp.get_validation_data(device)
        x_test, y_test = exp.get_test_data(device)
        x_test_ib, y_test_ib= exp.get_test_ib_data(device)

        logtarget = lambda theta : logposterior(theta, model, x_train, y_train, 0.1 )

        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('HyperNet_lat_dim', lat_dim)
        mlflow.log_param('HyperNet_nb_neurons', exp.param_count)

        Hyper_Nets=HyNetEns(ensemble_size,lat_dim, exp.param_count,activation,init_w,init_b).to(device)
        
        mlflow.log_param('learning_rate', learning_rate)
        optimizer = torch.optim.Adam(Hyper_Nets.parameters(), lr=learning_rate)
        
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)
        
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)

        mlflow.log_param('n_samples_KNN', n_samples_KNN)
        mlflow.log_param('n_samples_ED', n_samples_ED)
        mlflow.log_param('n_samples_LP', n_samples_LP)
        
        
        
        
        training_loss = []
        for t in range(max_iter):
            optimizer.zero_grad()


            ED=Hyper_Nets.KNNE(n_samples_KNN)
            LP=logtarget(Hyper_Nets(n_samples_LP)).mean()
            L =-ED-LP

            L.backward()

            #training_loss.append(L.detach().clone().cpu().numpy())

            
            lr = optimizer.param_groups[0]['lr']

            if t % 50 ==0:
                mlflow.log_metric("differential entropy", float(ED.detach().clone().cpu().numpy()),t)
                mlflow.log_metric("training loss", float(L.detach().clone().cpu().numpy()),t)
                mlflow.log_metric("learning rate", float(lr),t)
            
            scheduler.step(L.detach().clone().cpu().numpy())

            if verbose:
                stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
                print(stats)

            if lr < min_lr:
                break

            optimizer.step()

        with torch.no_grad():
            tempdir = tempfile.TemporaryDirectory()
            torch.save(Hyper_Nets,tempdir.name+'/hypernets.pt')
            mlflow.log_artifact(tempdir.name+'/hypernets.pt')

#            mlflow.log_metric("training loss", float(L.detach().clone().cpu().numpy()))
            
            pd.DataFrame(training_loss).to_csv(tempdir.name+'/training_loss.csv', index=False, header=False)
            mlflow.log_artifact(tempdir.name+'/training_loss.csv')
            nb_samples_postpred=int(np.ceil(10000/ensemble_size))
            logposteriorpredictive = exp.get_logposteriorpredictive_parallel_fn('cpu')
            train_post = logposteriorpredictive(Hyper_Nets(nb_samples_postpred).cpu(), model, x_train.cpu(), y_train.cpu(), 0.1)
            mlflow.log_metric("training log posterior predictive", -float(train_post.detach().cpu()))
            val_post = logposteriorpredictive(Hyper_Nets(nb_samples_postpred).detach().cpu(), model, x_validation.cpu(), y_validation.cpu(), 0.1)
            mlflow.log_metric("validation log posterior predictive", -float(val_post.detach().cpu()))
            test_post = logposteriorpredictive(Hyper_Nets(nb_samples_postpred).detach().cpu(), model, x_test.cpu(), y_test.cpu(), 0.1)
            mlflow.log_metric("test log posterior predictive", -float(test_post.detach().cpu()))
            test_ib_post = logposteriorpredictive(Hyper_Nets(nb_samples_postpred).detach().cpu(), model, x_test_ib.cpu(), y_test_ib.cpu(), 0.1)
            mlflow.log_metric("test in between log posterior predictive", -float(test_ib_post.detach().cpu()))

            
            x_lin =  torch.linspace(-2.,2.0).unsqueeze(1).cpu()
            nb_samples_plot=1000
            theta = Hyper_Nets.sample(nb_samples_plot).cpu()
            
            fig, ax = plt.subplots()
            fig.set_size_inches(11.7, 8.27)
            plt.xlim(-2, 2) 
            plt.ylim(-4, 4)
            plt.grid(True, which='major', linewidth=0.5)
            plt.title('Training set')
            plt.scatter(x_train.cpu(), y_train.cpu())
            for c in range(ensemble_size):
                for i in range(nb_samples_plot):
                    y_pred = model(theta[c,i].unsqueeze(0),x_lin.cpu())
                    plt.plot(x_lin, y_pred.squeeze(0), alpha=0.05, linewidth=1, color='C'+str(c+2))            
            fig.savefig(tempdir.name+'/training.png', dpi=4*fig.dpi)
            mlflow.log_artifact(tempdir.name+'/training.png')
            plt.close()

            fig, ax = plt.subplots()
            fig.set_size_inches(11.7, 8.27)
            plt.xlim(-2, 2) 
            plt.ylim(-4, 4)
            plt.grid(True, which='major', linewidth=0.5)
            plt.title('Validation set')
            plt.scatter(x_validation.cpu(), y_validation.cpu())
            for c in range(Hyper_Nets.nb_comp):
                for i in range(nb_samples_plot):
                    y_pred = model(theta[c,i].unsqueeze(0),x_lin).cpu()
                    plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='C'+str(c+2))             
            fig.savefig(tempdir.name+'/validation.png', dpi=4*fig.dpi)
            mlflow.log_artifact(tempdir.name+'/validation.png')
            plt.close()

            fig, ax = plt.subplots()
            fig.set_size_inches(11.7, 8.27)
            plt.xlim(-2, 2) 
            plt.ylim(-4, 4)
            plt.grid(True, which='major', linewidth=0.5)
            plt.title('Test set')
            plt.scatter(x_test.cpu(), y_test.cpu())
            for c in range(Hyper_Nets.nb_comp):
                for i in range(nb_samples_plot):
                    y_pred = model(theta[c,i].unsqueeze(0),x_lin).cpu()
                    plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='C'+str(c+2))             
            fig.savefig(tempdir.name+'/test.png', dpi=4*fig.dpi)
            mlflow.log_artifact(tempdir.name+'/test.png')
            plt.close()
            
            if ensemble_size>1:
                for c in range(ensemble_size):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(11.7, 8.27)
                    plt.xlim(-2, 2) 
                    plt.ylim(-4, 4)
                    plt.grid(True, which='major', linewidth=0.5)
                    plt.title('Test set (component '+str(c+1)+')')
                    plt.scatter(x_test.cpu(), y_test.cpu())                  
                    for i in range(nb_samples_plot):
                        y_pred = model(theta[c,i].unsqueeze(0),x_lin).cpu()
                        plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='C'+str(c+2))             
                    fig.savefig(tempdir.name+'/test'+str(c+1)+'.png', dpi=4*fig.dpi)
                    mlflow.log_artifact(tempdir.name+'/test'+str(c+1)+'.png')
                    plt.close()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="number of hypernets to train in the ensemble")
    parser.add_argument("--lat_dim", type=int, default=5,
                        help="number of latent dimensions of the hypernets")
    parser.add_argument("--init_w", type=float, default=0.1,
                        help="std for weight initialization of output layers")
    parser.add_argument("--init_b", type=float, default=0.0001,
                        help="std for bias initialization of output layers")    
    parser.add_argument("--n_samples_KNN", type=int, default=1000,
                        help="number of samples for KNN (nearest neighbour entropy estimate)")
    parser.add_argument("--n_samples_ED", type=int, default=5,
                        help="number of samples for MC estimation of differential entropy")
    parser.add_argument("--n_samples_LP", type=int, default=5,
                        help="number of samples for MC estimation of expected logposterior")
    parser.add_argument("--max_iter", type=int, default=100000,
                        help="maximum number of learning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--min_lr", type=float, default=0.0005,
                        help="minimum learning rate triggering the end of the optimization")
    parser.add_argument("--patience", type=int, default=100,
                        help="scheduler patience")
    parser.add_argument("--lr_decay", type=float, default=0.9,
                        help="scheduler multiplicative factor decreasing learning rate when patience reached")
    parser.add_argument("--device", type=str, default=None,
                        help="force device to be used")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="force device to be used")
    args = parser.parse_args()

    

 
    if args.device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device   
    
    print(args)
    
    main(args.ensemble_size,args.lat_dim,args.init_w,args.init_b, args.n_samples_KNN, args.n_samples_ED, args.n_samples_LP, args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, device=device, verbose=args.verbose)
    
