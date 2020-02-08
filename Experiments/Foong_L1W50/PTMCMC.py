import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from Tools.NNtools import *
import tempfile
import mlflow
import Experiments.Foong_L1W50.setup as exp
from Inference.ParallelTempering import *
import argparse
import pandas as pd


def main(numiter=1000, burnin=0, thinning=1, temperatures=[], maintempindex=0, baseMHproposalNoise=0.01, temperatureNoiseReductionFactor=0.5,
         std_init=0.01, optimize=0, seed=-1, device='cpu'):
    seeding(seed)
    
    xpname = exp.experiment_name +' PTMCMC'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(run_name='PTMCMC', experiment_id=expdata.experiment_id):
        mlflow.set_tag('device', device) 
        mlflow.set_tag('seed', seed)    
        logposterior = exp.get_logposterior_fn(device)
        x_train, y_train = exp.get_training_data(device)
        x_validation, y_validation = exp.get_validation_data(device)
        x_test, y_test = exp.get_test_data(device)
        logtarget = lambda theta: logposterior(theta, x_train, y_train, exp.sigma_noise)

        sampler = PTMCMCSampler(logtarget, exp.param_count, baseMHproposalNoise, temperatureNoiseReductionFactor, temperatures, device)

        sampler.initChains(nbiter=optimize, std_init=std_init)

        mlflow.log_param('numiter', numiter)
        mlflow.log_param('burnin', burnin)

        mlflow.log_param('thinning', thinning)
        mlflow.log_param('temperatures', temperatures)

        mlflow.log_param('optimize', optimize)
        mlflow.log_param('std_init', std_init)

        chains, ladderAcceptanceRate, swapAcceptanceRate, logProba = sampler.run(numiter)

        mlflow.set_tag('ladderAcceptanceRate', str(ladderAcceptanceRate))
        mlflow.set_tag('swapAcceptanceRate', str(swapAcceptanceRate))

        mlflow.set_tag('effective temperature', temperatures[maintempindex])
        ensemble = chains[maintempindex][burnin:-1:thinning]

        exp.log_model_evaluation(ensemble, device)


if __name__ == "__main__":
    # python -m Experiments.Foong_L1W50.PTMCMC --numiter=60000 --burnin=10000 --thinning=50 --temperatures=1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1 --maintempindex=0 --baseMHproposalNoise=0.006 --temperatureNoiseReductionFactor=0.5 --std_init=0.02 --optimize=2000000 --device=cpu
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
    parser.add_argument("--std_init", type=float, default=1.0,
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
        seed = np.random.randint(0, 2**31)
    else:
        seed = args.seed

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    temperatures = [float(n) for n in args.temperatures.split(',')]

    main(args.numiter, args.burnin, args.thinning, temperatures, args.maintempindex, args.baseMHproposalNoise,
         args.temperatureNoiseReductionFactor, args.std_init, args.optimize, seed=seed, device=device)
