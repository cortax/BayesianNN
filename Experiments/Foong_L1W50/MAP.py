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


def main(max_iter=100000, learning_rate=0.01, min_lr=0.0005, patience=100, lr_decay=0.9, gamma_alpha=1.0, gamma_beta=1.0, seed=-1, device='cpu'):
    seeding(seed)

    mlflow.set_experiment(exp.experiment_name)
    expdata = mlflow.get_experiment_by_name(exp.experiment_name)

    with mlflow.start_run(run_name='MAP', experiment_id=expdata.experiment_id):
        mlflow.set_tag('device', device) 
        mlflow.set_tag('seed', seed)    
        logposterior = exp.get_logposterior_fn(device)
        model = exp.get_model(device)
        x_train, y_train = exp.get_training_data(device)
        x_validation, y_validation = exp.get_validation_data(device)
        x_test, y_test = exp.get_test_data(device)
        logtarget = lambda theta : logposterior(theta, model, x_train, y_train, 0.1 )
        
        mlflow.log_param('gamma_alpha', gamma_alpha)
        mlflow.log_param('gamma_beta', gamma_beta)
        std = torch.distributions.Gamma(torch.tensor([gamma_alpha]), torch.tensor([gamma_beta])).sample()[0].float()
        theta = torch.nn.Parameter( torch.empty([1,exp.param_count],device=device).normal_(std=std), requires_grad=True)
        
        mlflow.log_param('learning_rate', learning_rate)
        optimizer = torch.optim.Adam([theta], lr=learning_rate)
        
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)
        
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)
        
        print(max_iter)
        
        for t in range(max_iter-1):
            optimizer.zero_grad()

            L = -torch.mean(logtarget(theta))
            L.backward()
            mlflow.log_metric("training log posterior", float(L.detach().cpu()), step=t)

            lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning rate", lr, step=t)

            scheduler.step(L.detach().clone().cpu().numpy())
            optimizer.step()

            if lr < min_lr:
                break

        with torch.no_grad():
            val_post = logposterior(theta, model, x_validation, y_validation, 0.1 )
            mlflow.log_metric("validation log posterior", -float(val_post.detach().cpu())/y_validation.shape[0])

            test_post = logposterior(theta, model, x_test, y_test, 0.1 )
            mlflow.log_metric("test log posterior", -float(test_post.detach().cpu())/y_test.shape[0])

            tempdir = tempfile.TemporaryDirectory()

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

            x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
            fig, ax = plt.subplots()
            fig.set_size_inches(11.7, 8.27)
            plt.xlim(-2, 2) 
            plt.ylim(-4, 4)
            plt.grid(True, which='major', linewidth=0.5)
            plt.title('Validation set')
            plt.scatter(x_validation.cpu(), y_validation.cpu())
            set_all_parameters(model, theta)
            y_pred = model(x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0, linewidth=1.0, color='black')
            res = 20
            for r in range(res):
                mass = 1.0-(r+1)/res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(), y_pred.squeeze(0).detach().cpu().numpy().squeeze()-3*0.1*((r+1)/res), y_pred.squeeze(0).detach().cpu().numpy().squeeze()+3*0.1*((r+1)/res), alpha=0.2*mass, color='lightblue')
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
            set_all_parameters(model, theta)
            y_pred = model(x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0, linewidth=1.0, color='black')
            res = 20
            for r in range(res):
                mass = 1.0-(r+1)/res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(), y_pred.squeeze(0).detach().cpu().numpy().squeeze()-3*0.1*((r+1)/res), y_pred.squeeze(0).detach().cpu().numpy().squeeze()+3*0.1*((r+1)/res), alpha=0.2*mass, color='lightblue')
            fig.savefig(tempdir.name+'/test.png', dpi=4*fig.dpi)
            mlflow.log_artifact(tempdir.name+'/test.png')
            plt.close()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--gamma_alpha", type=float, default=1.0,
                        help="parameter controling std~Gamma(alpha,beta) feeding theta~initialization(std)")
    parser.add_argument("--gamma_beta", type=float, default=1.0,
                        help="parameter controling std~Gamma(alpha,beta) feeding theta~initialization(std)")
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

    main(args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, args.gamma_alpha, args.gamma_beta, seed=seed, device=device)
