import torch
import argparse
import mlflow
import tempfile
from Experiments.foong import Setup
from Inference.MCMC import PTMCMCSampler


def learning(objective_fn, param_count, device, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize):
    sampler = PTMCMCSampler(
        objective_fn, param_count, baseMHproposalNoise, temperatureNoiseReductionFactor, temperatures, device)
        
    sampler.initChains(nbiter=optimize, std_init=std_init)

    chains, ladderAcceptanceRate, swapAcceptanceRate, logProba = sampler.run(numiter)
    ensemble = chains[maintempindex][burnin:-1:thinning]

    return ensemble, chains, ladderAcceptanceRate, swapAcceptanceRate, logProba

def log_exp_params(param_count, ladderAcceptanceRate, swapAcceptanceRate, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize, device='cpu'):

    mlflow.set_tag('device', device)
    mlflow.set_tag('dimensions', param_count)

    mlflow.set_tag('temperatures', temperatures)

    ladderAcceptanceRate=[float('{0:.2f}'.format(_)) for _ in ladderAcceptanceRate.tolist()]
    mlflow.set_tag('ladderAcceptanceRate', ladderAcceptanceRate)

    swapAcceptanceRate=[float('{0:.2f}'.format(_)) for _ in swapAcceptanceRate.tolist()]
    mlflow.set_tag('swapAcceptanceRate', swapAcceptanceRate)



    mlflow.log_param('numiter', numiter)
    mlflow.log_param('burnin', burnin)
    mlflow.log_param('thinning', thinning)
    mlflow.log_param('temperatureNoiseReductionFactor',temperatureNoiseReductionFactor)
    mlflow.log_param('maintempindex', maintempindex)
    mlflow.log_param('optimize',optimize)
    mlflow.log_param('baseMHproposalNoise', baseMHproposalNoise)
    mlflow.log_param('std_init',std_init)

def log_exp_metrics(evaluate_metrics, theta_ens, device):
    nLPP_train, nLPP_validation, nLPP_test, RSE_train, RSE_validation, RSE_test = evaluate_metrics(theta_ens, device)
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



def draw_experiment(makePlot, theta,device):
	fig = makePlot(theta,device)
	tempdir = tempfile.TemporaryDirectory()
	fig.savefig(tempdir.name + '/validation.png')
	mlflow.log_artifact(tempdir.name + '/validation.png')
	fig.close()

def PTMCMC(objective_fn, param_count, device, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize):
    ensemble = learning(objective_fn, param_count, device, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize)
    return ensemble



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
    #  python -m Experiments.foong.PTMCMC --numiter=10000 --burnin=100 --thinning=10 --temperatures=1.0,0.5,0.1 --maintempindex=0 --baseMHproposalNoise=0.01 --temperatureNoiseReductionFactor=0.5 --std_init=1.0 --optimize=0 --device=cpu
    #numiter as big as possible
    #burnin about 10% - 50%
    #thinning given by ensemble_size
    #ensemble size for metrics 10'000
    #ensemble_size for plotting
    #maintempindex index of temperature 1.0
    #baseMHproposalNoise
    #rule of thumb : look for acceptance rate = 40%-50% by tuning baseMHproposalNoise
    #temperatureNoiseReductionFactor: temperature factor for jump lenght in MCMC sampling ## DO NOT TOUCH!
    #temperatures= 1., .9 ,.8, .7, .6 , .5... to try



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
    theta_ens, _ ,  ladderAcceptanceRate, swapAcceptanceRate, _ =learning(setup.logposterior, setup.param_count, setup.device, args.numiter, args.burnin, args.thinning, temperatures, args.maintempindex, args.baseMHproposalNoise, args.temperatureNoiseReductionFactor, args.std_init, args.optimize)

    xpname = setup.experiment_name + '/PTMCMC'
    mlflow.set_experiment(xpname)
    expdata = mlflow.get_experiment_by_name(xpname)

    with mlflow.start_run(experiment_id=expdata.experiment_id):

        log_exp_params(setup.param_count, ladderAcceptanceRate, swapAcceptanceRate, args.numiter, args.burnin, args.thinning, temperatures, args.maintempindex, args.baseMHproposalNoise, args.temperatureNoiseReductionFactor, args.std_init, args.optimize, args.device)
        theta = torch.cat(theta_ens).cpu()
        log_exp_metrics(setup.evaluate_metrics,theta,'cpu')
        if setup.plot:
            theta=torch.cat(theta_ens[0:-1:10]).cpu()
            draw_experiment(setup.makePlot, theta,'cpu')