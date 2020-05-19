import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np

from tempfile import TemporaryDirectory

from Inference.FuNNeVI import FuNNeVI
from Models import BigGenerator
from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model

import tempfile

"""
grid search
lat_dim 5, 20

patience 100, 300

learning rate 0.01, 0.005
"""
## command line example
# python -m Experiments.GeNVI-pred --setup=boston --max_iter=10000 --learning_rate=0.05

def learning(loglikelihood, prior, projection, n_samples_FU, ratio_ood,
                   lat_dim, param_count, 
                   kNNE, n_samples_KL, n_samples_LL,  
                   max_iter, learning_rate, min_lr, patience, lr_decay, 
                   device, save_best):
    
	GeN = BigGenerator(lat_dim, param_count,device).to(device)
	#GeN = GeNetEns(ensemble_size, lat_dim, layerwidth, param_count, activation, init_w, init_b, device)

	with TemporaryDirectory() as temp_dir:
		optimizer = FuNNeVI(loglikelihood, prior, projection, n_samples_FU, ratio_ood,
                              kNNE, n_samples_KL, n_samples_LL, 
                              max_iter, learning_rate, min_lr, patience, lr_decay,
                              device, temp_dir, save_best=save_best)

		the_epoch, the_scores = optimizer.run(GeN)

	log_scores = [optimizer.score_elbo, optimizer.score_KL, optimizer.score_LL, optimizer.score_lr]
	return GeN, the_epoch, the_scores, log_scores


def log_GeNVI_experiment(setup,  n_samples_FU, ratio_ood, lat_dim, 
                         kNNE, n_samples_KL, n_samples_LL, 
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device, save_best):


    mlflow.set_tag('sigma_noise', setup.sigma_noise)    

    mlflow.set_tag('sigma_prior', setup.sigma_prior)    
    mlflow.set_tag('device', device)
    mlflow.set_tag('param_dim', setup.param_count)
    mlflow.set_tag('NNE', kNNE)
   
    mlflow.set_tag('device', device)
    mlflow.set_tag('save_best', save_best)

    mlflow.log_param('lat_dim', lat_dim)

    mlflow.log_param('n_samples_FU', n_samples_FU)
    mlflow.log_param('ratio_ood', ratio_ood)
    mlflow.log_param('n_samples_KL', n_samples_KL)
    mlflow.log_param('n_samples_LL', n_samples_LL)

    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('patience', patience)
    mlflow.log_param('lr_decay', lr_decay)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('min_lr', min_lr)


    


parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default=None,
                    help="data setup on which run the method")
parser.add_argument("--lat_dim", type=int, default=5,
                    help="number of latent dimensions of each hypernet")
parser.add_argument("--NNE", type=int, default=5,
                    help="k≥1 Nearest Neighbor Estimate")
parser.add_argument("--ratio_ood", type=float, default=.2,
                    help="ratio in [0,1] of ood inputs w.r.t data inputs for MC sampling of predictive distance")
parser.add_argument("--n_samples_FU", type=int, default=25,
                    help="number of samples for functions estimation")
parser.add_argument("--n_samples_KL", type=int, default=1000,
                    help="number of samples for NNE estimation of the KL")
parser.add_argument("--n_samples_LL", type=int, default=100,
                    help="number of samples for estimation of expected loglikelihood")
parser.add_argument("--max_iter", type=int, default=10000,
                    help="maximum number of learning iterations")
parser.add_argument("--learning_rate", type=float, default=0.002,
                    help="initial learning rate of the optimizer")
parser.add_argument("--min_lr", type=float, default=1e-7,
                    help="minimum learning rate triggering the end of the optimization")
parser.add_argument("--patience", type=int, default=400,
                    help="scheduler patience")
parser.add_argument("--lr_decay", type=float, default=.5,
                    help="scheduler multiplicative factor decreasing learning rate when patience reached")
parser.add_argument("--device", type=str, default=None,
                    help="force device to be used")
parser.add_argument('--save_best', dest='save_best', action='store_true',help="to return model with best ELBO during training, else return last model")
parser.set_defaults(save_best=False)
                    

if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    
    LPP=[]
    RMSE=[]
    times=[]
    setup_ = get_setup(args.setup)
    setup=setup_.Setup(args.device, 0) 

    xpname = setup.experiment_name + '/FuNNeVI'
    mlflow.set_experiment(xpname)
    with mlflow.start_run( ):
        
        log_GeNVI_experiment(setup, args.n_samples_FU, args.ratio_ood, args.lat_dim, 
                     args.NNE, args.n_samples_KL, args.n_samples_LL,
                     args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay,
                     args.device, args.save_best)
        
        for seed in range(10):
            setup_.Setup(args.device, seed) 
            
            loglikelihood=setup.loglikelihood
            projection=setup.projection
            size_sample=setup.n_train_samples
            param_count=setup.param_count

            #compute size of ood sample

            start = timeit.default_timer()

            def prior(n):
                return setup.sigma_prior*torch.randn(size=(n,param_count), device=args.device)


            GeN, the_epoch, the_scores, log_scores = learning(loglikelihood, prior, projection, args.n_samples_FU, args.ratio_ood,
                                                                    args.lat_dim, setup.param_count,
                                                                    args.NNE, args.n_samples_KL, args.n_samples_LL,
                                                                    args.max_iter, args.learning_rate, args.min_lr, args.patience,
                                                                    args.lr_decay, args.device, args.save_best)


            stop = timeit.default_timer()
            execution_time = stop - start

            times.append(execution_time)


            log_device = 'cpu'
            theta = GeN(10000).detach().to(log_device)
            
            
            with mlflow.start_run(run_name=str(seed),nested=True):

                for t in range(len(log_scores[0])):
                    mlflow.log_metric("elbo", float(log_scores[0][t]), step=100*t)
                    mlflow.log_metric("KL", float(log_scores[1][t]), step=100*t)
                    mlflow.log_metric("Exp. loglikelihood", float(log_scores[2][t]), step=100*t)
                    mlflow.log_metric("learning_rate", float(log_scores[3][t]), step=100*t)
                
                LPP_test, RMSE_test, _=log_exp_metrics(setup.evaluate_metrics, theta, execution_time, log_device)
                
                LPP.append(LPP_test[0].item())
                RMSE.append(RMSE_test[0].item())
                
                save_model(GeN)

                if setup.plot:
                    draw_experiment(setup, theta, log_device)

        mlflow.log_metric('RMSE_test', np.mean(RMSE))
        mlflow.log_metric('RMSE_test_std', np.std(RMSE))
        mlflow.log_metric('LPP_test', np.mean(LPP))
        mlflow.log_metric('LPP_test_std', np.std(LPP))
        mlflow.set_tag('time',np.mean(times))

