import argparse
import mlflow
import timeit
from tempfile import TemporaryDirectory
import torch


from Inference.Variational import MeanFieldVariationInference, MeanFieldVariationalDistribution
from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model



# CLI
# in root BayesianNN
# python -m Experiments.MFVI --max_iter=5000 --init_std=1. --learning_rate=0.1 --min_lr=0.00001 --patience=100 --lr_decay=0.5 --device=cuda:0 --verbose=1 --n_ELBO_samples=100 --setup=foong

g_lr=[0.01, 0.005, 0.001]#[0.01]#[0.01, 0.001, 0.0005]#[0.01, 0.005, 0.002, 0.001]
g_pat=[100, 200, 300]#[100, 300, 600]
lr_decay=0.5


def learning(objective_fn, max_iter, n_ELBO_samples, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device):

    mu=init_std*torch.randn(param_count, device=device)
    q0 = MeanFieldVariationalDistribution(param_count,mu=mu, sigma=0.0000001, device=device)

    with TemporaryDirectory() as temp_dir:
        optimizer = MeanFieldVariationInference(objective_fn, max_iter, n_ELBO_samples,
                                                learning_rate, min_lr, patience, lr_decay,  device, temp_dir)
        the_epoch, the_scores = optimizer.run(q0)

    log_scores = [optimizer.score_elbo, optimizer.score_entropy, optimizer.score_logposterior, optimizer.score_lr]
    return q0, the_epoch, the_scores, log_scores, optimizer.score_elbo[-1]

                   
def log_MFVI_experiment(setup, init_std,  n_ELBO_samples,
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device):

    mlflow.set_tag('lr grid', str(g_lr))
    mlflow.set_tag('patience grid', str(g_pat))
    
    mlflow.set_tag('device', device)
    mlflow.set_tag('sigma noise', setup.sigma_noise)
    mlflow.set_tag('dimensions', setup.param_count)

    mlflow.log_param('init_std', init_std)
    mlflow.log_param('n_ELBO_samples', n_ELBO_samples)

    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('min_lr', min_lr)
    mlflow.log_param('patience', patience)
    mlflow.log_param('lr_decay', lr_decay)
    return

def log_MFVI_run(the_epoch, the_scores, log_scores):    

    mlflow.log_metric('The epoch', the_epoch)

    mlflow.log_metric("The elbo", float(the_scores[0]))
    mlflow.log_metric("The entropy", float(the_scores[1]))
    mlflow.log_metric("The logposterior", float(the_scores[2]))

    for t in range(len(log_scores[0])):
        mlflow.log_metric("elbo", float(log_scores[0][t]), step=100*t)
        mlflow.log_metric("entropy", float(log_scores[1][t]), step=100*t)
        mlflow.log_metric("logposterior", float(log_scores[2][t]), step=100*t)
        mlflow.log_metric("learning_rate", float(log_scores[3][t]), step=100*t)
    return

def MFVI(setup, max_iter, n_ELBO_samples, init_std, min_lr, lr_decay):
    objective_fn = setup.logposterior
    param_count = setup.param_count
    device = setup.device
    


    best_elbo=torch.tensor(float('inf'))
    best_lr=None
    best_patience=None

    for lr in g_lr:
        for patience in g_pat:

            _, _, _, _, ELBO = learning(objective_fn, max_iter, n_ELBO_samples, lr, init_std, param_count, min_lr, patience, lr_decay, device)

            print('ELBO: '+str(ELBO.item()))
            if ELBO < best_elbo:
                best_elbo=ELBO
                best_lr=lr
                best_patience=patience
    
    xpname = setup.experiment_name + '/MFVI-gs'
    mlflow.set_experiment(xpname)

    with mlflow.start_run():
        log_MFVI_experiment(setup, init_std, n_ELBO_samples,
                            max_iter, best_lr, min_lr, best_patience, lr_decay,
                            device)

        
        for i in range(10):
            with mlflow.start_run(run_name=str(i),nested=True):

                start = timeit.default_timer()
                q, the_epoch, the_scores, log_scores, _ = learning(objective_fn, max_iter, n_ELBO_samples, best_lr, 
                                                                   init_std, param_count, min_lr, best_patience, 
                                                                   lr_decay, device)
                stop = timeit.default_timer()
                execution_time = stop - start


                log_MFVI_run(the_epoch, the_scores, log_scores)
                log_device='cpu'
                theta = q.sample(10000).detach().to(log_device)
                log_exp_metrics(setup.evaluate_metrics,theta,execution_time,log_device)


                save_model(q)

                if setup.plot:
                    draw_experiment(setup, theta[0:1000], log_device)


parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default=None,
                    help="data setup on which run the method")
parser.add_argument("--max_iter", type=int, default=15000, 
                    help="maximum number of learning iterations")
parser.add_argument("--min_lr", type=float, default=1e-7,
                    help="minimum learning rate triggering the end of the optimization")
parser.add_argument("--n_ELBO_samples", type=int, default=10,
                    help="number of Monte Carlo samples to compute ELBO")
parser.add_argument("--init_std", type=float, default=1.0,
                    help="parameter controling initialization of theta")
# parser.add_argument("--optimize", type=int, default=0,
#                     help="number of optimization iterations to initialize the state")
# parser.add_argument("--expansion", type=int, default=0,
#                     help="variational inference is done only on variance (0,1)")
parser.add_argument("--seed", type=int, default=None,
                    help="seed for random numbers")
parser.add_argument("--device", type=str, default=None,
                    help="force device to be used")

            
if __name__ == "__main__":


    
    args = parser.parse_args()
    print(args)

    
    setup_ = get_setup(args.setup)
    setup=setup_.Setup(args.device) 
    

                        
    MFVI(setup, args.max_iter, args.n_ELBO_samples,
         args.init_std, args.min_lr, lr_decay)


