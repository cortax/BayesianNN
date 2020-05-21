import torch

import numpy as np
import scipy.stats as st
from tqdm import tqdm, trange

from Inference.PointEstimate import AdamGradientDescent
from Inference.HMC import hamiltonian_monte_carlo_da

from numpy.linalg import norm

import argparse
import mlflow
import timeit

import arviz as az

#

from Experiments import log_exp_metrics, draw_experiment, get_setup, save_params_ens

"""
example command:

python -m Experiments.HMC --max_time= (en secondes) --setup=foong


python -m Experiments.HMC --setup=foong_mixed

python -m Experiments.HMC --setup=foong_sparse


python -m Experiments.HMC --setup=foong --numiter=2000 --burning=1000 --thinning=2 
python -m Experiments.HMC --setup=boston --numiter=2000 --burning=1000 --thinning=2 

"""

def _MAP(nbiter, std_init,logposterior, dim, device='cpu'):
        optimizer = AdamGradientDescent(logposterior, nbiter, .01, .00000001, 50, .5, device, True)

        theta0 = torch.empty((1, dim), device=device).normal_(0., std=std_init)
        best_theta, best_score, score = optimizer.run(theta0)

        return best_theta.detach().clone()


def log_exp_params(numiter, burning, thinning, initial_step_size):


    mlflow.log_param('numiter', numiter)
    mlflow.log_param('burning', burning)
    mlflow.log_param('thinning', thinning)
    mlflow.log_param('initial_step_size',initial_step_size)
    mlflow.log_param('path_len', path_len)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=str, default=None,
                        help="data setup on which run the method")
    parser.add_argument("--numiter", type=int, default=140000,
                        help="number of iterations in the Markov chain")
    parser.add_argument("--burning", type=int, default=40000,
                        help="number of initial samples to skip in the Markov chain")
    parser.add_argument("--thinning", type=int, default=10,
                        help="subsampling factor of the Markov chain")
    parser.add_argument("--step_size", type=float, default=0.002,
                        help="initial step_size for integrator")
    parser.add_argument("--path_len", type=int, default=100,
                        help="number of leapfrog integration steps")
    parser.add_argument("--optimize", type=int, default=5000,
                        help="number of optimization iterations to initialize the state")
    parser.add_argument("--device", type=str, default='cpu',
                        help="force device to be used")
    parser.add_argument("--max_time", type=float, default=float('inf'),
                        help="maximum time allocated")

    args = parser.parse_args()

    print(args)
    
    numiter_init=args.optimize
    numiter=args.numiter
    burning=args.burning
    thinning=args.thinning
    path_len=args.path_len
    initial_step_size=args.step_size
    
    setup_ = get_setup(args.setup)
    setup=setup_.Setup(args.device) 
    
    
    param_count=setup.param_count
    logposterior=setup.logposterior
    
    def potential(x):
        theta=torch.Tensor(x).requires_grad_(True).float()
        lp=logposterior(theta.unsqueeze(0))
        lp.backward()
        return -lp.detach().squeeze().numpy(), -theta.grad.numpy()

    
    start = timeit.default_timer()

    theta=_MAP(numiter_init, 1., logposterior, param_count)
    
        
    samples, scores = hamiltonian_monte_carlo_da(numiter, burning,thinning, potential, #
                                  initial_position=theta.squeeze().numpy(), 
                                  initial_step_size=initial_step_size,
                                  path_len=path_len, start_time=start, max_time=args.max_time)

    
    stop = timeit.default_timer()
    execution_time = stop - start
    
   
    theta=torch.as_tensor(samples)

    xpname = setup.experiment_name + '/HMC'
    mlflow.set_experiment(xpname)
    #mlflow logging
    with mlflow.start_run():


       

        torch.save(theta,'HMC_models/' + setup.experiment_name+'_'+mlflow.active_run().info.run_id+'.pt' )

        ##diagnostic with arviz

        data=az.convert_to_inference_data(theta.unsqueeze(0).numpy())

        ess_b=az.ess(data, method='bulk')
        ess_bulk=ess_b.to_dict()['data_vars']['x']['data']

        ess_t=az.ess(data, method='tail')
        ess_tail=ess_t.to_dict()['data_vars']['x']['data']


        trace=theta
        draws=trace.shape[0]    
        if draws % 2 ==1:
            trace=trace[0:-1]

        half_draws=int(trace.shape[0]/2)

        trace_0=trace[0:half_draws]
        trace_1=trace[half_draws:]
        folded_trace=np.stack([trace_0,trace_1])
        folded_data=az.convert_to_inference_data(folded_trace)

        folded_rhat_=az.rhat(folded_data)
        folded_rhat=folded_rhat_.to_dict()['data_vars']['x']['data']

        
        
        mlflow.set_tag('sigma_prior', setup.sigma_prior) 
        mlflow.set_tag('sigma_noise', setup.sigma_noise) 
        
        log_exp_params(numiter, burning, thinning, initial_step_size)
                
            
        for score in scores:
            for t in range(len(scores[score])):
                mlflow.log_metric(score, float(scores[score][t]), step=100*(t+1))
        
        for t in range(param_count):
            mlflow.log_metric('ess_bulk', float(ess_bulk[t]), step=t)
            mlflow.log_metric('ess_tail', float(ess_tail[t]), step=t)
            mlflow.log_metric('rhat 2-folded', float(folded_rhat[t]), step=t)

        mlflow.log_artifact('HMC_models/' + setup.experiment_name+'_'+mlflow.active_run().info.run_id+'.pt')
        
        log_exp_metrics(setup.evaluate_metrics, theta[0:1000], execution_time, 'cpu')        
                              
            
 