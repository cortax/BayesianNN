import torch
from torch import nn
import mlflow
import mlflow.pytorch
import tempfile
import argparse


from Inference.GeNVI_method import *
from Experiments.boston.setup import *
from Prediction.metrics import get_logposterior,RMSE, NLPD

splitting_index=0

init_b=.001
HN_activation=nn.ReLU()


KDE_prec=1.


 

def main(ensemble_size=1,lat_dim=5,HN_layerwidth=50,init_w=.2, n_samples_KDE=1000, n_samples_ED=50, n_samples_LP=50, max_iter=100000, learning_rate=0.03, min_lr=0.000001, patience=10, lr_decay=0.5,  device=None, verbose=False):
    
    xpname = experiment_name+'/GeNVI-KDE'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)
    
    with mlflow.start_run():#run_name='GeNVI-KDE', experiment_id=expdata.experiment_id
        
        X_train, y_train, y_train_un, X_test, y_test_un, inverse_scaler_y = get_data(splitting_index, device)
        
        mlflow.log_param('sigma noise', sigma_noise)
        mlflow.log_param('split', splitting_index)
        
        param_count, mlp=get_my_mlp()

        logtarget=get_logposterior(mlp,X_train,y_train,sigma_noise,device)
         
        KDE=get_KDE(device)
        
        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('HyperNet_lat_dim', lat_dim)
        mlflow.log_param('HyperNet_layerwidth', HN_layerwidth)
       
        Hyper_Nets=HyNetEns(ensemble_size, lat_dim, HN_layerwidth, param_count, HN_activation, init_w, init_b,device)
        
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)
        
        optimizer = torch.optim.Adam(Hyper_Nets.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)
        
        
        
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)
        

        mlflow.log_param('n_samples_KDE', n_samples_KDE)
        mlflow.log_param('n_samples_ED', n_samples_ED)
        mlflow.log_param('n_samples_LP', n_samples_LP)
        
        
        tempdir = tempfile.TemporaryDirectory()
        
        
        for t in range(max_iter):
            optimizer.zero_grad()

        #KDE:    
            ED=-KDE(Hyper_Nets(n_samples_ED),Hyper_Nets.sample(n_samples_KDE),KDE_prec).mean()
        #Nearest Neighbour Entropy estimation
            #ED=NNE(Hyper_Nets(n_samples_NNE))
            LP=logtarget(Hyper_Nets(n_samples_LP)).mean()
            L =-ED-LP
            L.backward()
            

            lr = optimizer.param_groups[0]['lr']
            
            mlflow.log_metric("ELBO", float(L.detach().clone().cpu().numpy()),t)
            mlflow.log_metric("-log posterior", float(-LP.detach().clone().cpu().numpy()),t)
            mlflow.log_metric("differential entropy", float(ED.detach().clone().cpu().numpy()),t)
            mlflow.log_metric("learning rate", float(lr),t)
            
            with torch.no_grad():
                theta=Hyper_Nets(100).detach()
                nlp_tr=NLPD(theta, mlp, X_train, y_train_un, sigma_noise, inverse_scaler_y, device)
                mlflow.log_metric("nlpd train", float(nlp_tr[1].detach().clone().cpu().numpy()),t)
                mlflow.log_metric("nlpd_std train", float(nlp_tr[0].detach().clone().cpu().numpy()),t)
                rms_tr=RMSE(theta,mlp,X_train,y_train_un,inverse_scaler_y,device)
                mlflow.log_metric("rmse train", float(rms_tr.detach().clone().cpu().numpy()),t)
                
                nlp=NLPD(theta,mlp,X_test, y_test_un, sigma_noise, inverse_scaler_y, device)              
                mlflow.log_metric("nlpd test", float(nlp[1].detach().clone().cpu().numpy()),t)
                mlflow.log_metric("nlpd_std test", float(nlp[0].detach().clone().cpu().numpy()),t)
                rms=RMSE(theta,mlp,X_test,y_test_un,inverse_scaler_y,device)
                mlflow.log_metric("rmse test", float(rms.detach().clone().cpu().numpy()),t)
                
                
                
            scheduler.step(L.detach().clone().cpu().numpy())

            if verbose:
                stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
                print(stats)

            if lr < min_lr:
                break
            
                
            optimizer.step()
            
        
        mlflow.pytorch.log_model(Hyper_Nets,'models')



if __name__== "__main__":
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
#    parser.add_argument("--KDE_prec", type=float, default=1.,
#                        help="factor reducing Silverman's bandwidth")
    parser.add_argument("--n_samples_KDE", type=int, default=1000,
                        help="number of samples for KDE")
    parser.add_argument("--n_samples_ED", type=int, default=50,
                        help="number of samples for MC estimation of differential entropy")
    parser.add_argument("--n_samples_LP", type=int, default=100,
                        help="number of samples for MC estimation of expected logposterior")
    parser.add_argument("--max_iter", type=int, default=1000000,
                        help="maximum number of learning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.03,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--min_lr", type=float, default=0.00000001,
                        help="minimum learning rate triggering the end of the optimization")
    parser.add_argument("--patience", type=int, default=10,
                        help="scheduler patience")
    parser.add_argument("--lr_decay", type=float, default=.1,
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
    
main(args.ensemble_size,args.lat_dim,args.layerwidth, args.init_w, args.n_samples_KDE, args.n_samples_ED, args.n_samples_LP, args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, device=device, verbose=args.verbose)

