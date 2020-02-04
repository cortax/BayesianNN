import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from Tools.NNtools import *
import tempfile
import mlflow
import Experiments.Foong_L1W50.setup as exp
from Inference.Variational import MeanFieldVariationalDistribution
from Inference.VariationalBoosting import MeanFieldVariationalMixtureDistribution
import argparse
import pandas as pd


def MAP(max_iter, init_std, learning_rate, patience, lr_decay, min_lr, logtarget, device, verbose):
    std = torch.tensor(init_std)
    theta = torch.nn.Parameter(torch.empty([1, exp.param_count], device=device).normal_(std=std), requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

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

def main(ensemble_size=1, max_iter=100000, learning_rate=0.01, min_lr=0.0005, n_ELBO_samples=50, patience=100, lr_decay=0.9, init_std=1.0, optimize=0, seed=-1, device='cpu', verbose=0):
    seeding(seed)

    xpname = exp.experiment_name + ' eMFVI'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(run_name='eMFVI', experiment_id=expdata.experiment_id):
        mlflow.set_tag('device', device)
        mlflow.set_tag('seed', seed)
        logposterior = exp.get_logposterior_fn(device)
        model = exp.get_model(device)
        x_train, y_train = exp.get_training_data(device)
        x_validation, y_validation = exp.get_validation_data(device)
        x_test, y_test = exp.get_test_data(device)
        logtarget = lambda theta: logposterior(theta, model, x_train, y_train, 0.1)

        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('learning_rate', learning_rate)

        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)

        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)

        mlflow.log_param('init_std', init_std)

        eMAP = [MAP(max_iter, init_std, learning_rate, patience, lr_decay, min_lr, logtarget, device, verbose) for _ in range(ensemble_size)]

        components = []
        for k in range(len(eMAP)):
            q_new = MeanFieldVariationalDistribution(exp.param_count, sigma=0.001, device=device)
            q_new.mu = nn.Parameter(eMAP[k].squeeze(0).to(device), requires_grad=True)
            components.append(q_new)

        with torch.no_grad():
            proportions = torch.ones([len(eMAP)], requires_grad=True, device=device) / len(eMAP)

        q = MeanFieldVariationalMixtureDistribution(proportions, components, device=device)
        [c.rho.detach_().requires_grad_(True) for c in q.components]
        [c.mu.detach_().requires_grad_(True) for c in q.components]

        optimizer = torch.optim.Adam([c.mu for c in q.components] + [c.rho for c in q.components], lr=learning_rate, betas=(0.999, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

        for t in range(max_iter):
            optimizer.zero_grad()

            Ln = []
            Z = q.sample(n_ELBO_samples)
            MU = torch.stack([c.mu for c in q.components])
            SIGMA = torch.stack([c.sigma for c in q.components])
            A_ = -0.5 * torch.log(2 * np.pi * SIGMA ** 2)
            B_ = (SIGMA ** 2)
            for j in range(n_ELBO_samples):
                z = Z[j, :].unsqueeze(0)
                # LQ = q.log_prob(z)

                P = A_ - (0.5 * (MU - z) ** 2) / B_
                LQ = torch.logsumexp(torch.log(q.proportions) + P.sum(dim=1), dim=0)

                LP = logposterior(z, model, x_train, y_train, sigma_noise=0.1)
                Ln.append(LQ - LP)

            L = torch.stack(Ln).mean()
            L.backward()

            learning_rate = optimizer.param_groups[0]['lr']
            scheduler.step(L.detach().clone().cpu().numpy())

            if verbose:
                stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, learning_rate)
                print(stats)

            if learning_rate < min_lr:
                break

            optimizer.step()

        ensemble = [q.sample() for _ in range(1000)]
        exp.log_model_evaluation(ensemble, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="number of model to train in the ensemble")
    parser.add_argument("--max_iter", type=int, default=100000,
                        help="maximum number of learning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--n_ELBO_samples", type=int, default=10,
                        help="number of Monte Carlo samples to compute ELBO")
    parser.add_argument("--min_lr", type=float, default=0.0005,
                        help="minimum learning rate triggering the end of the optimization")
    parser.add_argument("--patience", type=int, default=100,
                        help="scheduler patience")
    parser.add_argument("--lr_decay", type=float, default=0.9,
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
    args = parser.parse_args()

    print(args)

    if args.seed is None:
        seed = np.random.randint(0, 2 ** 31)
    else:
        seed = args.seed

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    main(args.ensemble_size, args.max_iter, args.learning_rate, args.min_lr, args.n_ELBO_samples, args.patience, args.lr_decay, args.init_std, args.optimize, seed, device, args.verbose)