import torch
import argparse
import mlflow
import tempfile
from Experiments.foong import Setup
from Inference.PointEstimate import AdamGradientDescent


def learning(objective_fn, max_iter, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose):
    optimizer = AdamGradientDescent(
        objective_fn, max_iter, learning_rate, min_lr, patience, lr_decay, device, verbose)

    theta0 = torch.empty((1, param_count), device=device).normal_(0., std=init_std)
    best_theta, best_score, score = optimizer.run(theta0)

    return best_theta, best_score, score


def log_MAP_experiment(setup, ensemble_best_theta, ensemble_best_score, ensemble_score, ensemble_size, max_iter, learning_rate, init_std, min_lr, patience, lr_decay, device):
    xpname = setup.experiment_name + '/MAP'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(experiment_id=expdata.experiment_id):
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

        for map in range(ensemble_size):
            with mlflow.start_run(run_name=str(map),nested=True):
                mlflow.log_metric("The logposterior", float(ensemble_best_score[map]))
                for t in range(len(ensemble_score[map])):
                    mlflow.log_metric("training loss", float(ensemble_score[map][t]),t)

        theta = torch.cat([torch.as_tensor(_) for _ in ensemble_best_theta])

        nLPP_train, nLPP_validation, nLPP_test, RSE_train, RSE_validation, RSE_test = setup.evaluate_metrics(theta)
        mlflow.log_metric("MnLPP_train", float(nLPP_train[0].cpu().numpy()))
        mlflow.log_metric("MnLPP_validation", float(nLPP_validation[0].cpu().numpy()))
        mlflow.log_metric("MnLPP_test", float(nLPP_test[0].cpu().numpy()))

        mlflow.log_metric("SnLPP_train", float(nLPP_train[1].cpu().numpy()))
        mlflow.log_metric("SnLPP_validation", float(nLPP_validation[1].cpu().numpy()))
        mlflow.log_metric("SnLPP_test", float(nLPP_test[1].cpu().numpy()))

        mlflow.log_metric("MSE_train", float(RSE_train[0].cpu().numpy()))
        mlflow.log_metric("MSE_validation", float(RSE_validation[0].cpu().numpy()))
        mlflow.log_metric("MSE_test", float(RSE_test[0].cpu().numpy()))

        mlflow.log_metric("SSE_train", float(RSE_train[1].cpu().numpy()))
        mlflow.log_metric("SSE_validation", float(RSE_validation[1].cpu().numpy()))
        mlflow.log_metric("SSE_test", float(RSE_test[1].cpu().numpy()))

        fig = setup.makeValidationPlot(theta)
        tempdir = tempfile.TemporaryDirectory()
        fig.savefig(tempdir.name+'/validation.png')
        mlflow.log_artifact(tempdir.name+'/validation.png')
        fig.close()

def MAP(setup, max_iter, learning_rate, init_std, min_lr, patience, lr_decay, device, verbose):
    objective_fn = setup.logposterior
    param_count = setup.param_count
    device = setup.device
    ensemble_size = 1
    best_theta, best_score, score = learning(objective_fn, max_iter, learning_rate, init_std, param_count, min_lr, patience,
                                             lr_decay, device, verbose)


    log_MAP_experiment(setup, [best_theta], [best_score], [score], ensemble_size, max_iter, learning_rate, init_std, min_lr, patience, lr_decay, device)

def eMAP(setup, ensemble_size, max_iter, learning_rate, init_std, min_lr, patience, lr_decay, device, verbose):
    objective_fn = setup.logposterior
    param_count = setup.param_count
    device = setup.device
    ensemble_best_theta = []
    ensemble_best_score = []
    ensemble_score = []
    for _ in range(ensemble_size):
        best_theta, best_score, score = learning(objective_fn, max_iter, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose)
        ensemble_best_theta.append(best_theta)
        ensemble_best_score.append(best_score)
        ensemble_score.append(score)

    log_MAP_experiment(setup, ensemble_best_theta, ensemble_best_score, ensemble_score, ensemble_size, max_iter, learning_rate, init_std, min_lr, patience, lr_decay, device)

if __name__ == "__main__":
    # example the commande de run 
    # python -m Experiments.foong.MAP --max_iter=10000 --init_std=0.1 --learning_rate=0.1 --min_lr=0.0001 --patience=100 --lr_decay=0.9 --device=cpu --verbose=1

    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="number of models to use in the ensemble")
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

    setup = Setup(args.device)

    if args.ensemble_size > 1:
        eMAP(setup, args.ensemble_size, args.max_iter, args.learning_rate, args.init_std, args.min_lr, args.patience, args.lr_decay, args.device, args.verbose)
    else:
        MAP(setup, args.max_iter, args.learning_rate, args.init_std, args.min_lr, args.patience, args.lr_decay, args.device, args.verbose)
    

    
