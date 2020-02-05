import torch
from torch import nn
import mlflow
import tempfile

from Inference.GeNVI_method import *
from Experiments.boston.setup import *
from Prediction.logposterior import get_logposterior

input_dim=13
nblayers = 1
activation_pn=nn.ReLU()
layerwidth = 50
splitting_index=0
sigma_noise

ensemble_size=1
lat_dim=5
HN_layerwidth=50
init_w=.2
init_b=.001
activation=nn.ReLU()


KDE_prec=1.
n_samples_KDE=1000
n_samples_NNE=1000
n_samples_ED=50
n_samples_LP=100
max_iter=10000000000
learning_rate=0.03
min_lr=0.0000005
patience=10
lr_decay=0.5
device='cpu'
verbose=True

 

def main():#(ensemble_size=1,lat_dim=5,init_w=.2,init_b=.001,KDE_prec=1.,n_samples_KDE=1000,n_samples_ED=20, n_samples_LP=20, max_iter=10000, learning_rate=0.001, min_lr=0.000005, patience=100, lr_decay=0.9,  device='cuda:1', verbose=True):
    
    xpname = 'Boston' +'GeNVI'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)
    
    with mlflow.start_run(run_name='GeNVI-KDE', experiment_id=expdata.experiment_id):
        
        X_train, y_train, X_test, y_test, inv_transform_y= get_data(splitting_index,device)
    
        param_count, mlp=get_mlp(input_dim,layerwidth,nblayers,activation)

        logtarget=get_logposterior(mlp,X_train,y_train,sigma_noise,device)

        Hyper_Nets=HyNetEns(ensemble_size, lat_dim, HN_layerwidth, param_count, activation, init_w, init_b,device)
        
        KDE=get_KDE(device)
        
        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('HyperNet_lat_dim', lat_dim)
        mlflow.log_param('HyperNet_nb_neurons', param_count)
       
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
        
        
        tempdir = tempfile.TemporaryDirectory()
        
        
        training_loss=[]
        for t in range(max_iter):
            optimizer.zero_grad()

        #KDE:    
            ED=-KDE(Hyper_Nets(n_samples_ED),Hyper_Nets.sample(n_samples_KDE),KDE_prec).mean()
        #Nearest Neighbour Entropy estimation
            #ED=NNE(Hyper_Nets(n_samples_NNE))
            LP=logtarget(Hyper_Nets(n_samples_LP)).mean()
            L =-ED-LP
        #    L=-logtarget(theta).squeeze()
            L.backward()
            
            #training_loss.append(L.detach().clone().cpu().numpy())

            training_loss.append(L.detach().clone().cpu().numpy())
            lr = optimizer.param_groups[0]['lr']

            
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
            
            
main()

"""            
        ensemble = [Hyper_Nets().detach().clone().cpu() for _ in range(1000)]
        exp.log_model_evaluation(ensemble, 'cpu')
       
        with torch.no_grad():
                      
            torch.save(Hyper_Nets,tempdir.name+'/hypernets.pt')
            mlflow.log_artifact(tempdir.name+'/hypernets.pt')

            
            x_lin =  torch.linspace(-2.,2.0).unsqueeze(1).cpu()
            nb_samples_plot=2000
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
                    y_pred = exp.mlp(x_lin.cpu(),theta[c,i].unsqueeze(0))
                    plt.plot(x_lin, y_pred.squeeze(0), alpha=0.05, linewidth=1, color='C'+str(c+2)) 
            fig.savefig(tempdir.name+'/trainingpc.png', dpi=5*fig.dpi)
            mlflow.log_artifact(tempdir.name+'/trainingpc.png')
            plt.close()
            
            if ensemble_size>1:
                for c in range(ensemble_size):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(11.7, 8.27)
                    plt.xlim(-2, 2) 
                    plt.ylim(-4, 4)
                    plt.grid(True, which='major', linewidth=0.5)
                    plt.title('Training set (component '+str(c+1)+')')                  
                    for i in range(nb_samples_plot):
                        y_pred = exp.mlp(x_lin.cpu(),theta[c,i].unsqueeze(0))
                        plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='C'+str(c+2))             
                    plt.scatter(x_train.cpu(), y_train.cpu())
                    fig.savefig(tempdir.name+'/training'+str(c)+'.png', dpi=5*fig.dpi)
                    mlflow.log_artifact(tempdir.name+'/training'+str(c)+'.png')
                    plt.close()
                    
                    
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="number of hypernets to train in the ensemble")
    parser.add_argument("--lat_dim", type=int, default=5,
                        help="number of latent dimensions of the hypernets")
    parser.add_argument("--init_w", type=float, default=0.2,
                        help="std for weight initialization of output layers")
    parser.add_argument("--init_b", type=float, default=0.000001,
                        help="std for bias initialization of output layers")    
    parser.add_argument("--KDE_prec", type=float, default=1.,
                        help="factor reducing Silverman's bandwidth")
    parser.add_argument("--n_samples_KDE", type=int, default=1000,
                        help="number of samples for KDE")
    parser.add_argument("--n_samples_ED", type=int, default=5,
                        help="number of samples for MC estimation of differential entropy")
    parser.add_argument("--n_samples_LP", type=int, default=5,
                        help="number of samples for MC estimation of expected logposterior")
    parser.add_argument("--max_iter", type=int, default=100000,
                        help="maximum number of learning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--min_lr", type=float, default=0.00001,
                        help="minimum learning rate triggering the end of the optimization")
    parser.add_argument("--patience", type=int, default=1000,
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
    
    main(args.ensemble_size,args.lat_dim,args.init_w,args.init_b, args.KDE_prec, args.n_samples_KDE, args.n_samples_ED, args.n_samples_LP, args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, device=device, verbose=args.verbose)
 
"""

