import torch
import argparse
import mlflow
from Experiments.boston import setup
from Inference.PointEstimate import AdamGradientDescent


def learning(objective_fn, max_iter, learning_rate, init_std, min_lr, patience, lr_decay, device, verbose):
    optimizer = AdamGradientDescent(
        objective_fn, max_iter, learning_rate, min_lr, patience, lr_decay, device, verbose)

    theta0 = torch.empty((1,setup.param_count), device=device).normal_(0., std=init_std)
    best_theta, best_score, score = optimizer.run(theta0)

    return best_theta, best_score, score


def log_experiment(best_theta, best_score, score, max_iter, learning_rate, init_std, min_lr, patience, lr_decay, device, verbose):

    xpname = setup.experiment_name + '/MAP'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)
    
    with mlflow.start_run(experiment_id=expdata.experiment_id): 
        mlflow.set_tag('device', device)
        mlflow.set_tag('sigma noise', setup.sigma_noise)
        mlflow.set_tag('dimensions', setup.param_count)

        mlflow.log_param('init_std', init_std)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('patience', patience)
        mlflow.log_param('lr_decay', lr_decay)
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('min_lr', min_lr)

        for t in range(len(score)):
            mlflow.log_metric("training loss", float(score[t]), step=t)



if __name__ == "__main__":
    # example the commande de run "python -m ... --param=1 ...

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_std", type=float, default=1.0,
                        help="parameter controling initialization of theta")
    parser.add_argument("--max_iter", type=int, default=100000,
                        help="maximum number of learning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.03,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--min_lr", type=float, default=0.00000001,
                        help="minimum learning rate triggering the end of the optimization")
    parser.add_argument("--patience", type=int, default=10,
                        help="scheduler patience")
    parser.add_argument("--lr_decay", type=float, default=.1,
                        help="scheduler multiplicative factor decreasing learning rate when patience reached")
    parser.add_argument("--device", type=str, default='cpu',
                        help="force device to be used")
    parser.add_argument("--verbose", type=int, default=0,
                        help="force device to be used")
    args = parser.parse_args()
    print(args)

    objective_fn = setup.get_logposterior(args.device)

    best_theta, best_score, score = learning(objective_fn, args.max_iter, args.learning_rate, args.init_std, args.min_lr,
                                             args.patience, args.lr_decay, args.device, args.verbose)

    log_experiment(best_theta, best_score, score, args.max_iter, args.learning_rate, args.init_std, args.min_lr,
                   args.patience, args.lr_decay, args.device, args.verbose)
