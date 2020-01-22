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




class HNet(nn.Module):
            def __init__(self, lat_dim, nb_neur, output_dim,  activation=nn.ReLU(), init_w=.4, init_b=0.001):
                super(HNet, self).__init__()
                self.lat_dim = lat_dim
                self.output_dim=output_dim
                self.hnet=nn.Sequential(
                        nn.Linear(lat_dim,nb_neur),
                        activation,
                        nn.Linear(nb_neur,output_dim)
                        )
                
                torch.nn.init.normal_(self.hnet[2].weight,mean=0., std=init_w)
                torch.nn.init.normal_(self.hnet[2].bias,mean=0., std=init_b)

    
            def forward(self, n=1):
                epsilon = torch.randn(size=(n,self.lat_dim))
                return self.hnet(epsilon)           

class HyNetEns(nn.Module):
    def __init__(self,nb_comp,lat_dim, output_dim, activation, init_w,init_b):
        super(HyNetEns, self).__init__()
        self.nb_comp=nb_comp
        self.output_dim=output_dim
        self.components= nn.ModuleList([HNet(lat_dim,output_dim,output_dim,activation,init_w,init_b) for i in range(nb_comp)])   

    # "Silverman's rule of thumb", Wand and Jones p.111 "Kernel Smoothing" 1995.                                 
    def get_H(self, nb_samples):
        theta=self.forward(nb_samples)
        c=torch.tensor(((nb_samples*(self.output_dim+2))/4)).pow(2/(self.output_dim+4))       
        H_=theta.var(0)/c
        #H_=theta.var(0).min(1).values/c*torch.ones(self.output_dim) #to try!
        return theta, H_.clamp(torch.finfo().eps,float('inf'))

    def KDE(self, theta_,theta, H):
        def kernel(theta1,theta2):
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(theta1, torch.diag(H))
            return mvn.log_prob(theta2)
        LQ=torch.Tensor(theta_.shape[0],theta.shape[0]) 
        for i in range(theta_.shape[0]):
            LQ[i]=kernel(theta_[i],theta) 
        return (LQ.logsumexp(1)-torch.log(torch.tensor(float(theta.shape[0])))).unsqueeze(1)   

    def sample(self, n=1):
        return torch.stack([self.components[c](n) for c in range(self.nb_comp)])

    
    def forward(self, n=1):
        return torch.cat([self.components[c](n).squeeze(0) for c in range(self.nb_comp)],dim=0)

    
    '''
    # "Silverman's rule of thumb", Wand and Jones p.111 "Kernel Smoothing" 1995.                                 
    def get_H(self, nb_samples, prec=KDE_prec):
        theta=self.sample(nb_samples)
        c=torch.tensor(((nb_samples*(self.output_dim+2))/4)).pow(2/(self.output_dim+4))*prec       
        H_=theta.var(1)/c
        #H_=theta.var(1).min(1).values/c*torch.ones(self.output_dim) #to try!
        return theta, H_.clamp(torch.finfo().eps,float('inf'))

    def KDE(self, theta_,theta, H_):
        def kernel(theta1,theta2,H):
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(theta1, torch.diag(H))
            return mvn.log_prob(theta2)
        LQ=torch.Tensor(theta_.shape[0],self.nb_comp,theta.shape[1]) 
        for c in range(self.nb_comp):
            for i in range(theta_.shape[0]):
                LQ[i,c]=kernel(theta_[i],theta[c],H_[c])
        return (LQ.logsumexp((1,2)).clamp(torch.finfo().min,float('inf'))-torch.log(torch.tensor(float(self.nb_comp*theta.shape[1])))).unsqueeze(1)
    '''
        
def main(ensemble_size=1,lat_dim=5,activation=nn.ReLU(),init_w=.15,init_b=.001,KDE_prec=1.,n_samples_KDE=1000,n_samples_ED=20, n_samples_LP=20, max_iter=10, learning_rate=0.001, min_lr=0.000001, patience=100, lr_decay=0.9,  device='cpu', verbose=True):
    
    xpname = exp.experiment_name + 'HyNet-KDE'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(run_name='HyNet-KDE', experiment_id=expdata.experiment_id):
        mlflow.set_tag('device', device) 
        logposterior = exp.get_logposterior_parallel_fn(device)#new!
        model = exp.get_parallel_model(device) #new!
        x_train, y_train = exp.get_training_data(device)
        x_validation, y_validation = exp.get_validation_data(device)
        x_test, y_test = exp.get_test_data(device)
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

        mlflow.log_param('KDE_prec', KDE_prec)
        mlflow.log_param('n_samples_KDE', n_samples_KDE)
        mlflow.log_param('n_samples_ED', n_samples_ED)
        mlflow.log_param('n_samples_LP', n_samples_LP)
        
        
        
        
        training_loss = []
        for t in range(max_iter):
            optimizer.zero_grad()

            theta,H=Hyper_Nets.get_H(n_samples_KDE)
            ED=-Hyper_Nets.KDE(Hyper_Nets(n_samples_ED),theta,1/KDE_prec*H).mean()
            LP=logtarget(Hyper_Nets(n_samples_LP)).mean()
            L =-ED-LP

            L.backward()

            training_loss.append(L.detach().clone().cpu().numpy())

            lr = optimizer.param_groups[0]['lr']
            scheduler.step(L.detach().clone().cpu().numpy())

            if verbose:
                stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
                print(stats)

            if lr < min_lr:
                break

            optimizer.step()

        with torch.no_grad():
            tempdir = tempfile.TemporaryDirectory()

            mlflow.log_metric("training loss", float(L.detach().clone().cpu().numpy()))
            
            pd.DataFrame(training_loss).to_csv(tempdir.name+'/training_loss.csv', index=False, header=False)
            mlflow.log_artifact(tempdir.name+'/training_loss.csv')

            logposteriorpredictive = exp.get_logposteriorpredictive_parallel_fn(device)
            train_post = logposteriorpredictive(Hyper_Nets(10000), model, x_train, y_train, 0.1)/len(y_train)
            mlflow.log_metric("training log posterior predictive", -float(train_post.detach().cpu()))
            val_post = logposteriorpredictive(Hyper_Nets(10000), model, x_validation, y_validation, 0.1)/len(y_validation)
            mlflow.log_metric("validation log posterior predictive", -float(val_post.detach().cpu()))
            test_post = logposteriorpredictive(Hyper_Nets(10000), model, x_test, y_test, 0.1)/len(y_test)
            mlflow.log_metric("test log posterior predictive", -float(test_post.detach().cpu()))
            
            
            x_lin =  torch.arange(-2.,2.0,0.01).unsqueeze(1).to(device)
            fig, ax = plt.subplots()
            fig.set_size_inches(11.7, 8.27)
            plt.xlim(-2, 2) 
            plt.ylim(-4, 4)
            plt.grid(True, which='major', linewidth=0.5)
            plt.title('Training set')
            plt.scatter(x_train.cpu(), y_train.cpu())
            
            theta = Hyper_Nets(100)
            for i in range(100):
                y_test = model(theta[i].unsqueeze(0),x_test)
                plt.plot(x_test.detach().cpu().numpy(), y_test.squeeze(0).cpu().numpy(), alpha=0.05, linewidth=1, color='green')
                #    plt.plot(x_test.cpu(), y_test.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='C'+str(c))
            fig.rcParams['agg.path.chunksize'] = 10000000000000000
            fig.savefig(tempdir.name+'/training.png', dpi=4*fig.dpi)
            mlflow.log_artifact(tempdir.name+'/training.png')
            plt.close()

            x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
            fig, ax = plt.subplots()
            fig.set_size_inches(11.7, 8.27)
            plt.xlim(-2, 2) 
            plt.ylim(-4, 4)
            plt.grid(True, which='major', linewidth=0.5)
            plt.title('Validation set')
            plt.scatter(x_validation.cpu(), y_validation.cpu())
            theta = Hyper_Nets.sample(100).detach()
            for c in range(Hyper_Nets.nb_comp):
                for i in range(100):
                    y_test = model(theta[c,i].unsqueeze(0),x_test)
                #    plt.plot(x_test.detach().cpu().numpy(), y_test.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='green')
                    plt.plot(x_test.cpu(), y_test.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='C'+str(c))           
            fig.savefig(tempdir.name+'/validation.png', dpi=4*fig.dpi)
            mlflow.log_artifact(tempdir.name+'/validation.png')
            plt.close()

            x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
            fig, ax = plt.subplots()
            fig.set_size_inches(11.7, 8.27)
            plt.xlim(-2, 2) 
            plt.ylim(-4, 4)
            plt.grid(True, which='major', linewidth=0.5)
            plt.title('Test set')
            plt.scatter(x_test.cpu(), y_test.cpu())
            theta = Hyper_Nets.sample(100).detach()
            for c in range(Hyper_Nets.nb_comp):
                for i in range(100):
                    y_test = model(theta[c,i].unsqueeze(0),x_test)
                #    plt.plot(x_test.detach().cpu().numpy(), y_test.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='green')
                    plt.plot(x_test.cpu(), y_test.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='C'+str(c))
            fig.savefig(tempdir.name+'/test.png', dpi=4*fig.dpi)
            mlflow.log_artifact(tempdir.name+'/test.png')
            plt.close()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="number of model to train in the ensemble")
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

    print(args)

 
    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
main()

#    main(20,nn.Tanh(), 1,5,1.,1000,100,100,args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, device=args.device, verbose=args.verbose)
