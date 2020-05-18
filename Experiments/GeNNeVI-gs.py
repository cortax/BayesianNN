import torch
from torch import nn
import argparse
import mlflow
import timeit

from tempfile import TemporaryDirectory

from Inference.GeNNeVI import GeNNeVI

from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model

from Models import BigGenerator

import tempfile


## command line example
# python -m Experiments.GeNNeVI --setup=foong --max_iter=20000 --learning_rate=0.05 --lat_dim= --layerwidth=

g_lr=[0.01, 0.005, 0.002, 0.001]#, 0.005, 0.001]#[0.01]#[0.01, 0.001, 0.0005]#[0.01, 0.005, 0.002, 0.001]
g_pat=[100, 250, 500]#, 400]#[100, 300, 600]
lr_decay=0.5

def GeNVI_run(objective_fn, lat_dim, param_count,
                   kNNE, n_samples_NNE, n_samples_LP,
                   max_iter, learning_rate, min_lr, patience, lr_decay,
                   device=None, save_best=False):
    
    GeN=BigGenerator(lat_dim=lat_dim,output_dim=param_count,device=device).to(device)
        

    with TemporaryDirectory() as temp_dir:
        model=GeNNeVI(objective_fn=logposterior,
                     kNNE=kNNE, n_samples_NNE=n_samples_NNE, n_samples_LP=n_samples_LP,
                     max_iter=max_iter, learning_rate=learning_rate, min_lr=min_lr, patience=patience, lr_decay=lr_decay,
                     device=device, temp_dir=temp_dir, save_best=save_best)
        the_epoch, the_elbo=model.run(GeN)

    ELBO=model.ELBO(GeN)
    return GeN, the_epoch, the_elbo, model.scores, ELBO.item()


def log_GeNVI_experiment(setup,
                         lat_dim, param_count,
                         kNNE, n_samples_NNE, n_samples_LP,
                         max_iter, learning_rate, min_lr, patience, lr_decay, device):

        
    mlflow.set_tag('lr grid', str(g_lr))
    mlflow.set_tag('patience grid', str(g_pat))
    
    mlflow.set_tag('sigma_prior', setup.sigma_prior)
    mlflow.set_tag('device', device)
    mlflow.set_tag('param_dim', setup.param_count)
    mlflow.set_tag('NNE', kNNE)
    mlflow.set_tag('lat_dim', lat_dim)


    mlflow.log_param('n_samples_NNE', n_samples_NNE)
    mlflow.log_param('n_samples_LP', n_samples_LP)

    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('patience', patience)
    mlflow.log_param('lr_decay', lr_decay)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('min_lr', min_lr)

def log_GeNVI_run(the_epoch, the_elbo, scores):
    
    mlflow.log_metric('Saved epoch', the_epoch)

    mlflow.log_metric("Saved elbo", the_elbo)

    for t in range(len(scores['ELBO'])):
        mlflow.log_metric("elbo", float(scores['ELBO'][t]), step=100*t)
        mlflow.log_metric("entropy", float(scores['Entropy'][t]), step=100*t)
        mlflow.log_metric("learning_rate", float(scores['lr'][t]), step=100*t)


parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default=None,
                    help="data setup on which run the method")
parser.add_argument("--lat_dim", type=int, default=5,
                    help="number of latent dimensions of each hypernet")
parser.add_argument("--kNNE", type=int, default=1,
                    help="k≥1 for k-Nearest Neighbor Estimate")
parser.add_argument("--n_samples_NNE", type=int, default=1000,
                    help="number of samples for NNE estimator")
parser.add_argument("--n_samples_LP", type=int, default=100,
                    help="number of samples for MC estimation of expected logposterior")
parser.add_argument("--max_iter", type=int, default=20000,
                    help="maximum number of learning iterations")
parser.add_argument("--learning_rate", type=float, default=0.01,
                    help="initial learning rate of the optimizer")
parser.add_argument("--min_lr", type=float, default=1e-7,
                    help="minimum learning rate triggering the end of the optimization")
parser.add_argument("--patience", type=int, default=300,
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

    setup_ = get_setup(args.setup)
    setup=setup_.Setup(args.device) 
    
    logposterior=setup.logposterior
    param_count=setup.param_count
    
    
    best_elbo=torch.tensor(float('inf'))
    best_lr=None
    best_patience=None
    for lr in g_lr:
        for patience in g_pat:

            _, _, _, _, ELBO =GeNVI_run(logposterior, 
                                        lat_dim=args.lat_dim, param_count=setup.param_count,
                                        kNNE=args.kNNE, n_samples_NNE=args.n_samples_NNE,
                                        n_samples_LP=args.n_samples_LP,
                                        max_iter=args.max_iter, learning_rate=lr, min_lr=args.min_lr,
                                        patience=patience, lr_decay=args.lr_decay,
                                        device=args.device,save_best=args.save_best)
            print('ELBO: '+str(ELBO))
            if ELBO < best_elbo:
                best_elbo=ELBO
                best_lr=lr
                best_patience=patience

    xpname = setup.experiment_name + '/GeNNeVI-gs'
    mlflow.set_experiment(xpname)

    with mlflow.start_run():
        log_GeNVI_experiment(setup,
                             args.lat_dim, setup.param_count,
                             args.kNNE, args.n_samples_NNE, args.n_samples_LP,
                             args.max_iter, best_lr, args.min_lr, best_patience, args.lr_decay,
                             args.device)   
        for i in range(10):
            with mlflow.start_run(run_name=str(i),nested=True):

                start = timeit.default_timer()

                GeN, the_epoch, the_elbo, scores, ELBO =GeNVI_run(logposterior, 
                                                                lat_dim=args.lat_dim, param_count=setup.param_count,
                                                                kNNE=args.kNNE, n_samples_NNE=args.n_samples_NNE,
                                                                n_samples_LP=args.n_samples_LP,
                                                                max_iter=args.max_iter, learning_rate=best_lr, min_lr=args.min_lr,
                                                                patience=best_patience, lr_decay=args.lr_decay,
                                                                device=args.device,save_best=args.save_best)



                stop = timeit.default_timer()
                time = stop - start


                #ML flow logging:#

                log_GeNVI_run(the_epoch, the_elbo, scores)

                log_device = 'cpu'
                theta = GeN(10000).detach().to(log_device)
                log_exp_metrics(setup.evaluate_metrics, theta, time, log_device)

                save_model(GeN)

                if setup.plot:
                    draw_experiment(setup, theta[0:1000], log_device)
