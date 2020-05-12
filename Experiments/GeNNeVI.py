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

def GeNVI_run(objective_fn, lat_dim, param_count,
                   kNNE, n_samples_NNE, n_samples_LP,
                   max_iter, learning_rate, min_lr, patience, lr_decay,
                   device=None):
    
    GeN=BigGenerator(lat_dim=lat_dim,output_dim=param_count,device=device).to(device)
        
    start = timeit.default_timer()
    
    
    with TemporaryDirectory() as temp_dir:
        model=GeNNeVI(objective_fn=logposterior,
                     kNNE=kNNE, n_samples_NNE=n_samples_NNE, n_samples_LP=n_samples_LP,
                     max_iter=max_iter, learning_rate=0.01, min_lr=0.000001, patience=300, lr_decay=0.5,
                     device=device, temp_dir=temp_dir, save_best=True)
        the_epoch, the_elbo=model.run(GeN)
    
    stop = timeit.default_timer()
    execution_time = stop - start

    return GeN, the_epoch, the_elbo, model.scores, execution_time


def log_GeNVI_experiment(setup, the_epoch, the_elbo, scores, time,
                         lat_dim, param_count,
                         kNNE, n_samples_NNE, n_samples_LP,
                         max_iter, learning_rate, min_lr, patience, lr_decay, device):

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
parser.add_argument("--max_iter", type=int, default=10000,
                    help="maximum number of learning iterations")
parser.add_argument("--learning_rate", type=float, default=0.01,
                    help="initial learning rate of the optimizer")
parser.add_argument("--min_lr", type=float, default=0.00000001,
                    help="minimum learning rate triggering the end of the optimization")
parser.add_argument("--patience", type=int, default=300,
                    help="scheduler patience")
parser.add_argument("--lr_decay", type=float, default=.5,
                    help="scheduler multiplicative factor decreasing learning rate when patience reached")
parser.add_argument("--device", type=str, default=None,
                    help="force device to be used")



if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    setup = get_setup(args.setup, args.device)

    logposterior=setup.logposterior
    param_count=setup.param_count

    
    GeN, the_epoch, the_elbo, scores, time=GeNVI_run(logposterior, 
                                                   lat_dim=args.lat_dim, param_count=setup.param_count,
                                                   kNNE=args.kNNE, n_samples_NNE=args.n_samples_NNE,
                                                   n_samples_LP=args.n_samples_LP,
                                                   max_iter=args.max_iter, learning_rate=args.learning_rate, min_lr=args.min_lr,
                                                   patience=args.patience, lr_decay=args.lr_decay, device=args.device)
    
    #ML flow logging:#
    
    xpname = setup.experiment_name + '/GeNNeVI'
    mlflow.set_experiment(xpname)

    with mlflow.start_run():
        log_GeNVI_experiment(setup, the_epoch, the_elbo, scores, time,
                             args.lat_dim, setup.param_count,
                             args.kNNE, args.n_samples_NNE, args.n_samples_LP,
                             args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay,
                             args.device)

        log_device = 'cpu'
        theta = GeN(1000).detach().to(log_device)
        log_exp_metrics(setup.evaluate_metrics, theta, time, log_device)

        save_model(GeN)

        if setup.plot:
            draw_experiment(setup.makePlot, theta, log_device)
