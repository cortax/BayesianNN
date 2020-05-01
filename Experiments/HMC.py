import torch

import numpy as np
import scipy.stats as st
from tqdm import tqdm, trange

from Inference.PointEstimate import AdamGradientDescent
from Inference.HMC import hamiltonian_monte_carlo_da

from Experiments.foong import Setup

from numpy.linalg import norm

import argparse
import mlflow
import timeit

from Experiments import log_exp_metrics, draw_experiment, get_setup, save_params_ens



def _MAP(nbiter, std_init,logposterior, dim, device='cpu'):
        optimizer = AdamGradientDescent(logposterior, nbiter, .01, .00000001, 50, .5, device, True)

        theta0 = torch.empty((1, dim), device=device).normal_(0., std=std_init)
        best_theta, best_score, score = optimizer.run(theta0)

        return best_theta.detach().clone()


def log_exp_params(numiter, burning, thinning, check_rate,initial_step_size):

    mlflow.log_param('numiter', numiter)
    mlflow.log_param('burning', burning)
    mlflow.log_param('thinning', thinning)
    mlflow.log_param('initial_step_size',initial_step_size)
    mlflow.log_param('path_len', path_len)
    mlflow.log_param('check_rate', check_rate)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=str, default=None,
                        help="data setup on which run the method")
    parser.add_argument("--numiter", type=int, default=50000,
                        help="number of iterations in the Markov chain")
    parser.add_argument("--burning", type=int, default=40000,
                        help="number of initial samples to skip in the Markov chain")
    parser.add_argument("--thinning", type=int, default=20,
                        help="subsampling factor of the Markov chain")
    parser.add_argument("--check_rate", type=int, default=500, 
                        help="check acceptance rate every check_rate steps for monitoring step_size")
    parser.add_argument("--step_size", type=float, default=0.002,
                        help="initial step_size for integrator")
    parser.add_argument("--path_len", type=int, default=100,
                        help="number of leapfrog integration steps")
    parser.add_argument("--optimize", type=int, default=1000,
                        help="number of optimization iterations to initialize the state")
    parser.add_argument("--device", type=str, default='cpu',
                        help="force device to be used")
    args = parser.parse_args()

    numiter_init=args.optimize
    numiter=args.numiter
    burning=args.burning
    thinning=args.thinning
    check_rate=args.check_rate
    path_len=args.path_len
    initial_step_size=args.step_size
    
    setup =get_setup(args.setup,args.device)

    
    param_count=setup.param_count
    logposterior=setup.logposterior
    
    def potential(x):
        theta=torch.Tensor(x).requires_grad_(True).float()
        lp=logposterior(theta.unsqueeze(0))
        lp.backward()
        return -lp.detach().squeeze().numpy(), -theta.grad.numpy()

    
    start = timeit.default_timer()

    theta=_MAP(numiter_init,1., logposterior, param_count)

    
    samples, rates, step_sizes, log_prob = hamiltonian_monte_carlo_da(numiter, burning,thinning, potential, #
                                  initial_position=theta.squeeze().numpy(), 
                                  #check_rate=check_rate,
                                  initial_step_size=initial_step_size,
                                  path_len=path_len
                                    )

    stop = timeit.default_timer()
    execution_time = stop - start
    
    results=[samples, rates, step_sizes,log_prob]
    theta=torch.as_tensor(samples)
    
    xpname = setup.experiment_name + '/HMC'
    mlflow.set_experiment(xpname)

    with mlflow.start_run():

        log_exp_params(numiter, burning, thinning, check_rate,initial_step_size)
        
        log_exp_metrics(setup.evaluate_metrics,theta,execution_time,'cpu')
    
        for t in range(len(rates)):
            mlflow.log_metric("acceptance", float(rates[t]), step=check_rate*(t+1))
            mlflow.log_metric("step_sizes", float(step_sizes[t]), step=check_rate*(t+1))
            mlflow.log_metric("log_prob", float(log_prob[t]), step=check_rate*(t+1))

          
        if setup.plot:
            draw_experiment(setup.makePlot, theta, 'cpu')
        #
        save_params_ens(theta)
    

