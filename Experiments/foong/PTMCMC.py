import torch
import argparse
import mlflow
import tempfile
from Experiments.foong import Setup
from Inference.MCMC import PTMCMCSampler


def learning(objective_fn, param_count, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize, device):
    sampler = PTMCMCSampler(
        objective_fn, param_count, baseMHproposalNoise, temperatureNoiseReductionFactor, temperatures, device)
        
    sampler.initChains(nbiter=optimize, std_init=std_init)

    chains, ladderAcceptanceRate, swapAcceptanceRate, logProba = sampler.run(numiter)
    ensemble = chains[maintempindex][burnin:-1:thinning]

    return ensemble, chains, ladderAcceptanceRate, swapAcceptanceRate, logProba
                   
def log_experiment(setup, best_theta, best_score, score, ensemble_size, max_iter, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose, nested=False):
    xpname = setup.experiment_name + '/MAP'
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

        avgNLL_train, avgNLL_validation, avgNLL_test = setup.evaluate_metrics(theta)
        mlflow.log_metric("avgNLL_train", float(avgNLL_train.cpu().numpy()))
        mlflow.log_metric("avgNLL_validation", float(avgNLL_validation.cpu().numpy()))
        mlflow.log_metric("avgNLL_test", float(avgNLL_test.cpu().numpy()))

        fig = setup.makeValidationPlot(theta)
        tempdir = tempfile.TemporaryDirectory()
        fig.savefig(tempdir.name+'/validation.png')
        mlflow.log_artifact(tempdir.name+'/validation.png')
        fig.close()

def PTMCMC(setup, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize):
    objective_fn = setup.logposterior
    param_count = setup.param_count
    device = setup.device

    q = learning(objective_fn, param_count, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize, device)
    
    #log_experiment(setup, best_theta, best_score, score, ensemble_size, max_iter, learning_rate, init_std, param_count, min_lr, patience, lr_decay, device, verbose, nested=False)

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
    #  python -m Experiments.foong.PTMCMC --numiter=100 --burnin=10 --thinning=2 --temperatures=1.0,0.5,0.1 --maintempindex=0 --baseMHproposalNoise=0.01 --temperatureNoiseReductionFactor=0.5 --std_init=1.0 --optimize=0 --device=cpu

    parser = argparse.ArgumentParser()
    parser.add_argument("--numiter", type=int, default=1000,
                        help="number of iterations in the Markov chain")
    parser.add_argument("--burnin", type=int, default=0,
                        help="number of initial samples to skip in the Markov chain")
    parser.add_argument("--thinning", type=int, default=1,
                        help="subsampling factor of the Markov chain")
    parser.add_argument("--temperatures", type=str, default=None,
                        help="temperature ladder in the form [t0, t1, t2, t3]")
    parser.add_argument("--maintempindex", type=int, default=None,
                        help="index of the temperature to use to make the chain (ex: 0 for t0)")
    parser.add_argument("--baseMHproposalNoise", type=float, default=0.01,
                        help="standard-deviation of the isotropic proposal")
    parser.add_argument("--temperatureNoiseReductionFactor", type=float, default=0.5,
                        help="factor adapting the noise to the corresponding temperature")
    parser.add_argument("--std_init", type=float, default=1.0,
                        help="parameter controling initialization of theta")
    parser.add_argument("--optimize", type=int, default=0,
                        help="number of optimization iterations to initialize the state")
    parser.add_argument("--seed", type=int, default=None,
                        help="value insuring reproducibility")
    parser.add_argument("--device", type=str, default=None,
                        help="force device to be used")
    args = parser.parse_args()
    print(args)

    setup = Setup(args.device)

    temperatures = [float(n) for n in args.temperatures.split(',')]

    PTMCMC(setup, args.numiter, args.burnin, args.thinning, temperatures, args.maintempindex, args.baseMHproposalNoise, args.temperatureNoiseReductionFactor, args.std_init, args.optimize)
