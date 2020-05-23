import torch
from torch import nn
import argparse
import mlflow
import timeit


from tempfile import TemporaryDirectory

from Inference.FuNNeVI import FuNNeVI
from Models import BigGenerator, SLPGenerator
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

def learning(loglikelihood, batch, size_data, prior, projection, n_samples_FU, ratio_ood, p,
                   lat_dim, param_count, 
                   kNNE, n_samples_KL, n_samples_LL,  
                   max_iter, learning_rate, min_lr, patience, lr_decay, 
                   device):

    GeN = BigGenerator(lat_dim, param_count,device).to(device)
    #GeN=SLPGenerator(lat_dim, param_count,device).to(device)
    #GeN = GeNetEns(ensemble_size, lat_dim, layerwidth, param_count, activation, init_w, init_b, device)

    with TemporaryDirectory() as temp_dir:
        optimizer = FuNNeVI(loglikelihood, batch, size_data, prior, projection, n_samples_FU, ratio_ood, p,
                              kNNE, n_samples_KL, n_samples_LL, 
                              max_iter, learning_rate, min_lr, patience, lr_decay,
                              device, temp_dir)

        ELBO = optimizer.run(GeN)
    
    return GeN, optimizer.scores, ELBO.item()


def log_GeNVI_experiment(ELBO, setup,  n_samples_FU, ratio_ood, scores, p, batch,
                         lat_dim, 
                         kNNE, n_samples_KL, n_samples_LL, 
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device):


    mlflow.set_tag('batch_size', batch)

    mlflow.set_tag('sigma_noise', setup.sigma_noise)    

    mlflow.set_tag('sigma_prior', setup.sigma_prior)    
    mlflow.set_tag('device', device)
    mlflow.set_tag('param_dim', setup.param_count)
    mlflow.set_tag('NNE', kNNE)
   
    mlflow.set_tag('device', device)

    mlflow.log_param('lat_dim', lat_dim)

    mlflow.log_param('L_p norm', p)
    mlflow.log_param('n_samples_FU', n_samples_FU)
    mlflow.log_param('ratio_ood', ratio_ood)
    mlflow.log_param('n_samples_KL', n_samples_KL)
    mlflow.log_param('n_samples_LL', n_samples_LL)

    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('patience', patience)
    mlflow.log_param('lr_decay', lr_decay)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('min_lr', min_lr)


    mlflow.log_metric("The elbo", ELBO)

    for t in range(len(scores['ELBO'])):
        mlflow.log_metric("elbo", float(scores['ELBO'][t]), step=100*t)
        mlflow.log_metric("KL", float(scores['KL'][t]), step=100*t)
        mlflow.log_metric("LL", float(scores['LL'][t]), step=100*t)        
        mlflow.log_metric("learning_rate", float(scores['lr'][t]), step=100*t)




parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default=None,
                    help="data setup on which run the method")
parser.add_argument("--lat_dim", type=int, default=5,
                    help="number of latent dimensions of each hypernet")
parser.add_argument("--NNE", type=int, default=1,
                    help="k≥1 Nearest Neighbor Estimate")
parser.add_argument("--ratio_ood", type=float, default=1.,
                    help="ratio in [0,1] of ood inputs w.r.t data inputs for MC sampling of predictive distance")
parser.add_argument("--n_samples_FU", type=int, default=20,
                    help="number of samples for functions estimation")
parser.add_argument("--p_norm", type=int, default=2,
                    help="L_p norm to use for functions")
parser.add_argument("--n_samples_KL", type=int, default=1000,
                    help="number of samples for NNE estimation of the KL")
parser.add_argument("--n_samples_LL", type=int, default=100,
                    help="number of samples for estimation of expected loglikelihood")
parser.add_argument("--batch", type=int, default=None,
                    help="size of batches for likelihood evaluation")
parser.add_argument("--max_iter", type=int, default=20000,
                    help="maximum number of learning iterations")
parser.add_argument("--learning_rate", type=float, default=0.001,
                    help="initial learning rate of the optimizer")
parser.add_argument("--min_lr", type=float, default=1e-9,
                    help="minimum learning rate triggering the end of the optimization")
parser.add_argument("--patience", type=int, default=600,
                    help="scheduler patience")
parser.add_argument("--lr_decay", type=float, default=.5,
                    help="scheduler multiplicative factor decreasing learning rate when patience reached")
parser.add_argument("--device", type=str, default=None,
                    help="force device to be used")

                    
if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    setup_ = get_setup(args.setup)
    setup=setup_.Setup(args.device) 
    
    loglikelihood=setup.loglikelihood
    projection=setup.projection
    size_sample=setup.n_train_samples
    param_count=setup.param_count
    
    batch=args.batch
    if batch is None:
        batch=size_sample

    #compute size of ood sample
    
    start = timeit.default_timer()

    def prior(n):
        return setup.sigma_prior*torch.randn(size=(n,param_count), device=args.device)


    GeN, log_scores, ELBO = learning(loglikelihood, batch, size_sample, prior, 
                                                      projection, args.n_samples_FU, args.ratio_ood, args.p_norm,
                                                      args.lat_dim, setup.param_count,
                                                      args.NNE, args.n_samples_KL, args.n_samples_LL,
                                                      args.max_iter, args.learning_rate, args.min_lr, args.patience,
                                                      args.lr_decay, args.device)

    
    stop = timeit.default_timer()
    execution_time = stop - start

    xpname = setup.experiment_name + '/FuNNeVI'
    mlflow.set_experiment(xpname)

    with mlflow.start_run():
        save_model(GeN)

        log_GeNVI_experiment(ELBO, setup, args.n_samples_FU, args.ratio_ood, log_scores, args.p_norm, batch,
                             args.lat_dim, 
                             args.NNE, args.n_samples_KL, args.n_samples_LL,
                             args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay,
                             args.device)

        log_device = 'cpu'
        
        theta = GeN(1000).detach().to(log_device)
        log_exp_metrics(setup.evaluate_metrics, theta, execution_time, log_device)


        if setup.plot:
            draw_experiment(setup, theta[0:1000], log_device)
    