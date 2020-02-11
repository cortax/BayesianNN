import torch
import argparse
import mlflow
import tempfile
from Experiments.foong import Setup
from Inference.Variational import MeanFieldVariationInference, MeanFieldVariationalDistribution


def learning(objective_fn, max_iter, n_ELBO_samples, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose):
    optimizer = MeanFieldVariationInference(
        objective_fn, max_iter, n_ELBO_samples, learning_rate, min_lr, patience, lr_decay, device, verbose)

    q0 = MeanFieldVariationalDistribution(setup.param_count, sigma=0.0000001, device=setup.device)
    q = optimizer.run(q0)
    return q
                   
def log_experiment(setup, best_theta, best_score, score, ensemble_size, max_iter, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose, nested=False):
    xpname = setup.experiment_name + '/MFVI'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)
    
    with mlflow.start_run(experiment_id=expdata.experiment_id, nested=nested): 
        mlflow.set_tag('device', device)
        mlflow.set_tag('sigma noise', setup.sigma_noise)
        mlflow.set_tag('dimensions', setup.param_count)

        mlflow.log_param('ensemble_size', ensemble_size)
        mlflow.log_param('init_std', init_std)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)

        if best_score is not None:
            mlflow.log_metric("training loss", float(best_score))
        if score is not None:
            for t in range(len(score)):
                mlflow.log_metric("training loss", float(score[t]), step=t)

        if type(best_theta) is list:
            theta = torch.cat([torch.tensor(a) for a in best_theta])
        else:
            theta = torch.tensor(best_theta)

        nLPP_train, nLPP_validation, nLPP_test, RSE_train, RSE_validation, RSE_test = setup.evaluate_metrics(theta)
        mlflow.log_metric("MnLPP_train", float(nLPP_train[0].cpu().numpy()))
        mlflow.log_metric("MnLPP_validation", float(nLPP_validation[0].cpu().numpy()))
        mlflow.log_metric("MnLPP_test", float(nLPP_test[0].cpu().numpy()))

        mlflow.log_metric("MRSE_train", float(RSE_train[0].cpu().numpy()))
        mlflow.log_metric("MRSE_validation", float(RSE_validation[0].cpu().numpy()))
        mlflow.log_metric("MRSE_test", float(RSE_test[0].cpu().numpy()))

        fig = setup.makeValidationPlot(theta)
        tempdir = tempfile.TemporaryDirectory()
        fig.savefig(tempdir.name+'/validation.png')
        mlflow.log_artifact(tempdir.name+'/validation.png')
        fig.close()

def MFVI(setup, max_iter, n_ELBO_samples, learning_rate, init_std, min_lr, patience, lr_decay, device, verbose):
    objective_fn = setup.logposterior
    param_count = setup.param_count
    device = setup.device
    ensemble_size = 1
    q = learning(objective_fn, max_iter, n_ELBO_samples, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose)
    
    #log_experiment(setup, q, best_score, score, ensemble_size, max_iter, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose, nested=False)

# def eMFVI(setup, ensemble_size, max_iter, learning_rate, init_std, min_lr, patience, lr_decay, device, verbose):
#     objective_fn = setup.logposterior
#     param_count = setup.param_count
#     device = setup.device
#     ensemble_best_theta = []
#     ensemble_best_score = []
#     ensemble_score = []
#     for _ in range(ensemble_size):
#         best_theta, best_score, score = learning(objective_fn, max_iter, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose)
#         ensemble_best_theta.append(best_theta)
#         ensemble_best_score.append(best_score)
#         ensemble_score.append(score)

#     log_experiment(setup, ensemble_best_theta, None, None, ensemble_size, max_iter, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose, True)

if __name__ == "__main__":
    # example the commande de run 
    # python -m Experiments.foong.MFVI --ensemble_size=1 --max_iter=100 --init_std=0.1 --learning_rate=0.1 --min_lr=0.0001 --patience=100 --lr_decay=0.9 --device=cuda:0 --verbose=1 --n_ELBO_samples=100

    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="number of models to use in the ensemble")
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

    setup = Setup(args.device)

    if args.ensemble_size > 1:
        pass
        #eMFVI(setup, args.ensemble_size, args.max_iter, args.learning_rate, args.init_std, args.min_lr, args.patience, args.lr_decay, args.device, args.verbose)
    else:
        MFVI(setup, args.max_iter, args.n_ELBO_samples, args.learning_rate, args.init_std, args.min_lr, args.patience, args.lr_decay, args.device, args.verbose)
    