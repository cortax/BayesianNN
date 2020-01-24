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


def main(ensemble_size=1, max_iter=100000, learning_rate=0.01, min_lr=0.0005, patience=100, lr_decay=0.9, init_std=1.0, seed=-1, device='cpu'):
    seeding(seed)
    
    xpname = exp.experiment_name +' eMAP'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(run_name='eMAP', experiment_id=expdata.experiment_id):
        mlflow.set_tag('device', device) 
        mlflow.set_tag('seed', seed)    
        logposterior = exp.get_logposterior_fn(device)
        model = exp.get_model(device)
        x_train, y_train = exp.get_training_data(device)
        x_validation, y_validation = exp.get_validation_data(device)
        x_test, y_test = exp.get_test_data(device)
        logtarget = lambda theta : logposterior(theta, model, x_train, y_train, 0.1 )

        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('learning_rate', learning_rate)

        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)

        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)

        ensemble = []
        for k in range(ensemble_size):
            with mlflow.start_run(run_name='component', nested=True):
                mlflow.log_param('init_std', init_std) 
                std = torch.tensor(init_std)
                theta = torch.nn.Parameter( torch.empty([1,exp.param_count],device=device).normal_(std=std), requires_grad=True)
                optimizer = torch.optim.Adam([theta], lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

                training_loss = []
                for t in range(max_iter-1):
                    optimizer.zero_grad()

                    L = -torch.mean(logtarget(theta))
                    L.backward()

                    training_loss.append(L.detach().clone().cpu().numpy())

                    lr = optimizer.param_groups[0]['lr']
                    scheduler.step(L.detach().clone().cpu().numpy())
                    if lr < min_lr:
                        break

                    optimizer.step()
                        
                with torch.no_grad():  
                    tempdir = tempfile.TemporaryDirectory()

                    mlflow.log_metric("training loss", float(L.detach().clone().cpu().numpy()))

                    pd.DataFrame(training_loss).to_csv(tempdir.name+'/training_loss.csv', index=False, header=False)
                    mlflow.log_artifact(tempdir.name+'/training_loss.csv')

                    logposteriorpredictive = exp.get_logposteriorpredictive_fn(device)
                    train_post = logposteriorpredictive([theta], model, x_train, y_train, 0.1)/len(y_train)
                    mlflow.log_metric("training log posterior predictive", -float(train_post.detach().cpu()))
                    val_post = logposteriorpredictive([theta], model, x_validation, y_validation, 0.1)/len(y_validation)
                    mlflow.log_metric("validation log posterior predictive", -float(val_post.detach().cpu()))
                    test_post = logposteriorpredictive([theta], model, x_test, y_test, 0.1)/len(y_test)
                    mlflow.log_metric("test log posterior predictive", -float(test_post.detach().cpu()))

                    x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
                    fig, ax = plt.subplots()
                    fig.set_size_inches(11.7, 8.27)
                    plt.xlim(-2, 2) 
                    plt.ylim(-4, 4)
                    plt.grid(True, which='major', linewidth=0.5)
                    plt.title('Training set')
                    plt.scatter(x_train.cpu(), y_train.cpu())
                    set_all_parameters(model, theta)
                    y_pred = model(x_lin)
                    plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0, linewidth=1.0, color='black')
                    res = 20
                    for r in range(res):
                        mass = 1.0-(r+1)/res
                        plt.fill_between(x_lin.detach().cpu().numpy().squeeze(), y_pred.squeeze(0).detach().cpu().numpy().squeeze()-3*0.1*((r+1)/res), y_pred.squeeze(0).detach().cpu().numpy().squeeze()+3*0.1*((r+1)/res), alpha=0.2*mass, color='lightblue')
                    fig.savefig(tempdir.name+'/training.png', dpi=4*fig.dpi)
                    mlflow.log_artifact(tempdir.name+'/training.png')
                    plt.close()

                    ensemble.append(theta.detach().clone())

        exp.log_model_evaluation(ensemble, device)

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
    parser.add_argument("--init_std", type=float, default=1.0,
                        help="parameter controling initialization of theta")
    parser.add_argument("--seed", type=int, default=None,
                        help="scheduler patience")
    parser.add_argument("--device", type=str, default=None,
                        help="force device to be used")
    args = parser.parse_args()

    print(args)

    if args.seed is None:
        seed = np.random.randint(0,2**31)
    else:
        seed = args.seed

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    main(args.ensemble_size, args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, args.init_std, seed=seed, device=device)
