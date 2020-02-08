import torch
from torch import nn
import mlflow
import mlflow.pytorch
import tempfile
import argparse


from Prediction.metrics import get_logposterior, log_metrics, seeding
from Inference.MFVI_method import MeanFieldVariationalDistribution, MeanFieldVariationalMixtureDistribution


from Experiments.foong.setup import *



"""
def MAP(dim,max_iter, init_std, learning_rate, patience, lr_decay, min_lr, logtarget, device, verbose):
    std = torch.tensor(init_std)
    theta = torch.nn.Parameter(torch.empty([1, dim], device=device).normal_(std=std), requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)#, patience=patience, factor=lr_decay)
    
    for t in range(max_iter):
        optimizer.zero_grad()

        L = -torch.mean(logtarget(theta))
        L.backward()

        lr = optimizer.param_groups[0]['lr']
        scheduler.step(L.detach().clone().cpu().numpy())

        if verbose:
            stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
            print(stats)

        if lr < min_lr:
            break
        optimizer.step()
    return theta.detach().clone()
"""

def main(ensemble_size=1, max_iter=100000, learning_rate=0.01, min_lr=0.0005,  n_samples_ED=50, n_samples_LP=50, patience=100, lr_decay=0.9, init_std=1.0, optimize=0, seed=-1, device='cpu', verbose=0, show_metrics=False):
    seeding(seed)

    xpname = experiment_name + '/MFVI'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run():#run_name='eMFVI', experiment_id=expdata.experiment_id
        mlflow.set_tag('device', device)
        mlflow.set_tag('seed', seed)
        
        X_train, y_train, X_test, y_test, X_ib_test, y_ib_test, X_valid, y_valid, inverse_scaler_y = get_data(device)
        
        mlflow.log_param('sigma noise', sigma_noise)

        param_count, mlp=get_my_mlp()
        mlflow.set_tag('dimensions', param_count)

        logtarget=get_logposterior(mlp,X_train,y_train,sigma_noise,device)

        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('learning_rate', learning_rate)

        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)

        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)

        mlflow.log_param('init_std', init_std)

        MFVI=MeanFieldVariationalMixtureDistribution(ensemble_size,param_count,init_std,device=device)
        

        optimizer = torch.optim.Adam(MFVI.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

        for t in range(max_iter):
            optimizer.zero_grad()

            ED=-MFVI.log_prob(MFVI(n_samples_ED)).mean()
            LP=logtarget(MFVI(n_samples_LP)).mean()
            L =-ED-LP
            L.backward()

            lr = optimizer.param_groups[0]['lr']
            scheduler.step(L.detach().clone().cpu().numpy())
            
            mlflow.log_metric("ELBO", float(L.detach().squeeze().clone().cpu().numpy()),t)
            mlflow.log_metric("-log posterior", float(-LP.detach().squeeze().clone().cpu().numpy()),t)
            mlflow.log_metric("differential entropy", float(ED.detach().clone().cpu().numpy()),t)
            mlflow.log_metric("learning rate", float(lr),t)
            mlflow.log_metric("epoch", t)
            
            if show_metrics:
                with torch.no_grad():
                    theta = MFVI(100)
                    log_metrics(theta, mlp, X_train, y_train, X_test, y_test, sigma_noise, inverse_scaler_y, t,device)

            if verbose:
                stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
                print(stats)

            if lr < min_lr:
                break

            optimizer.step()
        
        
        with torch.no_grad():
            theta = MFVI(1000)
            log_metrics(theta, mlp, X_train, y_train, X_test, y_test, sigma_noise, inverse_scaler_y, t,device)

            theta_plot=MFVI.sample(100)
            plot_test(X_train, y_train, X_test,y_test,theta_plot, mlp)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="number of model to train in the ensemble")
    parser.add_argument("--max_iter", type=int, default=100000,
                        help="maximum number of learning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--n_samples_ED", type=int, default=50,
                        help="number of samples for MC estimation of differential entropy")
    parser.add_argument("--n_samples_LP", type=int, default=100,
                        help="number of samples for MC estimation of expected logposterior")   
    parser.add_argument("--min_lr", type=float, default=0.0005,
                        help="minimum learning rate triggering the end of the optimization")
    parser.add_argument("--patience", type=int, default=10,
                        help="scheduler patience")
    parser.add_argument("--lr_decay", type=float, default=0.1,
                        help="scheduler multiplicative factor decreasing learning rate when patience reached")
    parser.add_argument("--init_std", type=float, default=1.0,
                        help="parameter controling initialization of theta")
    parser.add_argument("--optimize", type=int, default=0,
                        help="number of optimization iterations to initialize the state")
    parser.add_argument("--expansion", type=int, default=0,
                        help="variational inference is done only on variance (0,1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="scheduler patience")
    parser.add_argument("--device", type=str, default=None,
                        help="force device to be used")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="force device to be used")
    parser.add_argument("--show_metrics", type=bool, default=False,
                        help="log metrics during training")        
    args = parser.parse_args()

    print(args)

    if args.seed is None:
        seed = np.random.randint(0, 2 ** 31)
    else:
        seed = args.seed

    if args.device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    print(args)
    
    main(args.ensemble_size, args.max_iter, args.learning_rate, args.min_lr,  args.n_samples_ED, args.n_samples_LP,  args.patience, args.lr_decay, args.init_std, args.optimize, seed, device, args.verbose, args.show_metrics)
