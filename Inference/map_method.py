import numpy as np
import torch
from torch import nn
import math
import argparse
import mlflow
import mlflow.pytorch

from Prediction.metrics import get_logposterior, log_metrics

# 1) Ajouter l'EarlyStopping (validation) ou le rollback best (training). En RAM .to('cpu')
# 2) Vérifier la performance de logger les métriques avec et sans log interne dans la boucle
# 3) Faire un train(70)-validation(30)

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="number of MAPs to train in the ensemble")
    parser.add_argument("--init_std", type=float, default=1.0,
                        help="parameter controling initialization of theta")
    parser.add_argument("--max_iter", type=int, default=100000,
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
    return parser

def main(get_data, get_model, sigma_noise, experiment_name, ensemble_size, init_std, max_iter, learning_rate, min_lr, patience, lr_decay, device, verbose):
    xpname = experiment_name + '/MAP'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)
    
    with mlflow.start_run(): #run_name='GeNVI-KDE', experiment_id=expdata.experiment_id
        mlflow.set_tag('device', device)
        
        param_count, mlp=get_model()
        
        mlflow.log_param('sigma noise', sigma_noise)
        mlflow.set_tag('dimensions', param_count)

        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('init_std', init_std)

        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)

        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)
        
        with mlflow.start_run(run_name='MAP'):
            X_train, y_train, y_train_un, X_test, y_test_un, inverse_scaler_y = get_data(device)
            logtarget=get_logposterior(mlp, X_train, y_train, sigma_noise, device)

            theta_ens = torch.empty((ensemble_size,param_count), device=device)

            for k in range(ensemble_size):
                with mlflow.start_run(run_name='component', nested=True):
                    theta = torch.empty((1,param_count), device=device)
                    torch.nn.init.normal_(theta, 0., =init_std)
                    theta.requires_grad=True

                    optimizer = torch.optim.Adam([theta], lr=learning_rate)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

                    for t in range(max_iter-1):
                        optimizer.zero_grad()

                        L = -torch.mean(logtarget(theta))
                        L.backward()

                        lr = optimizer.param_groups[0]['lr']

                        # TODO: Vérifier la performance ici avec et sans log interne dans la boucle
                        mlflow.log_metric("-log posterior", float(L.detach().clone().cpu().numpy()), t)
                        mlflow.log_metric("learning rate", float(lr), t)
                        mlflow.log_metric("epoch", t)

                        if verbose:
                            stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(k, t, max_iter, L, lr)
                            print(stats)

                        scheduler.step(L.detach().clone().cpu().numpy())
                        optimizer.step()

                        if lr < min_lr:
                            break

                    with torch.no_grad():
                        log_metrics(theta.detach(), mlp, X_train, y_train_un, X_test, y_test_un, sigma_noise, inverse_scaler_y, t, device)
                        theta_ens[k]=theta.squeeze().detach()

            with torch.no_grad():
                theta = theta_ens
                log_metrics(theta, mlp, X_train, y_train_un, X_test, y_test_un, sigma_noise, inverse_scaler_y, t, device)

