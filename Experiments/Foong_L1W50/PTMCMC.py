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


def main(numiter=1000, burnin=0, thinning=1, temperatures=[1.0], maintempindex=0, baseMHproposalNoise=0.01, temperatureNoiseReductionFactor=0.5, \
         init_std=0.01, optimize=0, seed=seed, device=device):
    seeding(seed)
    
    xpname = exp.experiment_name +' PTMCMC'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(run_name='PTMCMC', experiment_id=expdata.experiment_id):
        mlflow.set_tag('device', device) 
        mlflow.set_tag('seed', seed)    
        logposterior = exp.get_logposterior_fn(device)
        model = exp.get_model(device)
        x_train, y_train = exp.get_training_data(device)
        x_validation, y_validation = exp.get_validation_data(device)
        x_test, y_test = exp.get_test_data(device)
        logtarget = lambda theta : logposterior(theta, model, x_train, y_train, 0.1 )

        sampler = PTMCMCSampler(logtarget, exp.param_count, baseMHproposalNoise, temperatureNoiseReductionFactor, temperatures, device)

        sampler.initChains(nbiter=optimize, init_std=init_std)

        mlflow.log_param('numiter', numiter)
        mlflow.log_param('burnin', burnin)

        mlflow.log_param('thinning', thinning)
        mlflow.log_param('temperatures', temperatures)

        mlflow.log_param('optimize', optimize)
        mlflow.log_param('init_std', init_std)

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
                
        with torch.no_grad():
            tempdir = tempfile.TemporaryDirectory()

            logposteriorpredictive = exp.get_logposteriorpredictive_fn(device)
            train_post = logposteriorpredictive(ensemble, model, x_train, y_train, 0.1)/len(y_train)
            mlflow.log_metric("training log posterior predictive", -float(train_post.detach().cpu()))
            val_post = logposteriorpredictive(ensemble, model, x_validation, y_validation, 0.1)/len(y_validation)
            mlflow.log_metric("validation log posterior predictive", -float(val_post.detach().cpu()))
            test_post = logposteriorpredictive(ensemble, model, x_test, y_test, 0.1)/len(y_test)
            mlflow.log_metric("test log posterior predictive", -float(test_post.detach().cpu()))

            x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
            fig, ax = plt.subplots()
            fig.set_size_inches(11.7, 8.27)
            plt.xlim(-2, 2) 
            plt.ylim(-4, 4)
            plt.grid(True, which='major', linewidth=0.5)
            plt.title('Training set')
            plt.scatter(x_train.cpu(), y_train.cpu())
            for theta in ensemble:
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
            for theta in ensemble:
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
            for theta in ensemble:
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

    parser.add_argument("--numiter", type=int, default=1000,
                        help="number of iterations in the Markov chain")
    parser.add_argument("--burnin", type=int, default=0,
                        help="number of initial samples to skip in the Markov chain")
    parser.add_argument("--thinning", type=int, default=1,
                        help="subsampling factor of the Markov chain")
    parser.add_argument("--temperatures", type=str, default=None,
                        help="temperature ladder in the form [t0, t1, t2, t3]")
    parser.add_argument("--maintempindex", type=int, default=None,
                        help="index of the temperature to use to make the chain (ex: 0 for t0)")
    parser.add_argument("--baseMHproposalNoise", type=float, default=0.01,
                        help="standard-deviation of the isotropic proposal")
    parser.add_argument("--temperatureNoiseReductionFactor", type=float, default=0.5,
                        help="factor adapting the noise to the corresponding temperature")
    parser.add_argument("--init_std", type=float, default=1.0,
                        help="parameter controling initialization of theta")
    parser.add_argument("--optimize", type=int, default=0,
                        help="number of optimization iterations to initialize the state")
    parser.add_argument("--seed", type=int, default=None,
                        help="value insuring reproducibility")
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

    main(args.numiter, args.burnin, args.thinning, args.temperatures, args.maintempindex, args.baseMHproposalNoise, \
         args.temperatureNoiseReductionFactor, args.init_std, args.optimize, seed=seed, device=device)
