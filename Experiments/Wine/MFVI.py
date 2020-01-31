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
import argparse


def main(max_iter=100000, learning_rate=0.01, min_lr=0.0005, n_ELBO_samples=100, patience=100, lr_decay=0.9, init_std=1.0, optimize=0, expansion=0, seed=-1, device='cpu', verbose=0):
    seeding(seed)

    xpname = exp.experiment_name + ' MFVI'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(run_name='MFVI', experiment_id=expdata.experiment_id):
        mlflow.set_tag('device', device)
        mlflow.set_tag('seed', seed)
        logposterior = exp.get_logposterior_fn(device)
        x_train, y_train = exp.get_training_data(device)
        x_validation, y_validation = exp.get_validation_data(device)
        x_test, y_test = exp.get_test_data(device)
        logtarget = lambda theta: logposterior(theta, x_train, y_train, exp.sigma_noise)

        mlflow.log_param('n_ELBO_samples', n_ELBO_samples)
        mlflow.log_param('learning_rate', learning_rate)

        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)

        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)

        mlflow.log_param('init_std', init_std)
        mlflow.log_param('optimize', optimize)

        std = torch.tensor(init_std)
        theta = torch.nn.Parameter(torch.empty([1, exp.param_count], device=device).normal_(std=std), requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

        for t in range(optimize):
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

        q = MeanFieldVariationalDistribution(exp.param_count, sigma=0.0000001, device=device)
        q.mu = nn.Parameter(theta.detach().clone(), requires_grad=True)
        q.rho.requires_grad = True
        if expansion:
            q.mu.requires_grad = False
            mlflow.log_param('expansion', 1)
        else:
            q.mu.requires_grad = True
            mlflow.log_param('expansion', 0)
        optimizer = torch.optim.Adam(q.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)

        training_loss = []
        for t in range(max_iter - 1):
            optimizer.zero_grad()

            z = q.sample(n_ELBO_samples)
            LQ = q.log_prob(z)
            LP = logposterior(z, x_train, y_train, sigma_noise=0.1)
            L = (LQ - LP).mean()

            L.backward()

            if verbose:
                stats = 'Epoch [{}/{}], Training Loss: {}, Learning Rate: {}'.format(t, max_iter, L, lr)
                print(stats)

            training_loss.append(L.detach().clone().cpu().numpy())

            lr = optimizer.param_groups[0]['lr']
            scheduler.step(L.detach().clone().cpu().numpy())
            if lr < min_lr:
                break

            optimizer.step()

        ensemble = [q.sample() for _ in range(1000)]
        exp.log_model_evaluation(ensemble, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_iter", type=int, default=100000,
                        help="maximum number of learning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--min_lr", type=float, default=0.0005,
                        help="minimum learning rate triggering the end of the optimization")
    parser.add_argument("--n_ELBO_samples", type=int, default=10,
                        help="number of Monte Carlo samples to compute ELBO")
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
                        help="seed for random numbers")
    parser.add_argument("--device", type=str, default=None,
                        help="force device to be used")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="force device to be used")
    args = parser.parse_args()

    print(args)

    if args.seed is None:
        seed = np.random.randint(0, 2 ** 31)
    else:
        seed = int(args.seed)

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    main(args.max_iter, args.learning_rate, args.min_lr, args.n_ELBO_samples, args.patience, args.lr_decay, args.init_std, args.optimize, args.expansion, seed, device, args.verbose)
