import torch
import argparse
import mlflow
import timeit


from Inference.MCMC import PTMCMCSampler

from Experiments import log_exp_metrics, draw_experiment, get_setup, save_params_ens


def learning(objective_fn, param_count, device, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize):
    sampler = PTMCMCSampler(
        objective_fn, param_count, baseMHproposalNoise, temperatureNoiseReductionFactor, temperatures, device)
        
    sampler.initChains(nbiter=optimize, std_init=std_init)

    ensemble, ladderAcceptanceRate, swapAcceptanceRate = sampler.run(numiter,burnin,thinning)


    return ensemble, ladderAcceptanceRate, swapAcceptanceRate

def log_exp_params(param_count, len_theta, ladderAcceptanceRate, swapAcceptanceRate, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize, device='cpu'):

    mlflow.set_tag('device', device)
    mlflow.set_tag('dimensions', param_count)

    mlflow.set_tag('temperatures', temperatures)
    mlflow.set_tag('nb_theta', len_theta)


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



def PTMCMC(objective_fn, param_count, device, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize):
    ensemble = learning(objective_fn, param_count, device, numiter, burnin, thinning, temperatures, maintempindex, baseMHproposalNoise, temperatureNoiseReductionFactor, std_init, optimize)
    return ensemble



if __name__ == "__main__":
    # example the commande de run
    # python -m Experiments.PTMCMC --numiter=20000 --baseMHproposalNoise=0.01 --optimize=10000 --setup=foong
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
    parser.add_argument("--setup", type=str, default=None,
                        help="data setup on which run the method")
    parser.add_argument("--numiter", type=int, default=100000,
                        help="number of iterations in the Markov chain")
    parser.add_argument("--burnin", type=int, default=None,
                        help="number of initial samples to skip in the Markov chain")
    parser.add_argument("--thinning", type=int, default=None,
                        help="subsampling factor of the Markov chain")
    parser.add_argument("--temperatures", type=str, default='1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.85, 1.95, 2.15, 2.3, 2.6, 3., 3.2, 3.6, 4., 4.4, 4.8, 5.5, 6.4, 7.7, 9., 11., 13.7, 17.5, 24., 35., 48.0, 81.2, 137.7, 233.6, 396.8, 674.1, 1145.6',
                        #'1.0, 1.4, 2., 3., 4.6, ,7.2 ,  6.5, 8.5 , 10.3, 14.0, 19.0, 26., 35., 48.0, 81.2, 137.7, 233.6, 396.8, 674.1, 1145.6',
                        #'1.0, 1.3, 1.8,| 2.6, 3.9,| 5.5,| 7.2,| 10.3,| 14.0, 19.0,| 26., 35., 48.0, 81.2, 137.7, 233.6, 396.8, 674.1, 1145.6'
                        #Fibonacci '1.0, 2.0, 3.0, 5.0, 7.0, 12.0, 19.0, 31.0, 50.0, 81.0, 131.0, 212.0, 343.0, 555.0',
                        help="temperature ladder in the form t0, t1, t2, t3")
    parser.add_argument("--maintempindex", type=int, default=0,
                        help="index of the temperature to use to make the chain (ex: 0 for t0)")
    parser.add_argument("--baseMHproposalNoise", type=float, default=0.0019,
                        help="standard-deviation of the isotropic proposal")
    parser.add_argument("--temperatureNoiseReductionFactor", type=float, default=0.5,
                        help="factor adapting the noise to the corresponding temperature")
    parser.add_argument("--std_init", type=float, default=1.0,
                        help="parameter controling initialization of theta")
    parser.add_argument("--optimize", type=int, default=20000,
                        help="number of optimization iterations to initialize the state")
    parser.add_argument("--seed", type=int, default=None,
                        help="value insuring reproducibility")
    parser.add_argument("--device", type=str, default=None,
                        help="force device to be used")
    args = parser.parse_args()

    if args.burnin is None:
        args.burnin =int(0.1*args.numiter)


    theta_ens_size=1000
    if args.thinning is None:
        args.thinning=max(1,int((args.numiter-args.burnin)/theta_ens_size))
        #numiter-burnin=thinning theta_ens_size

    print(args)

    setup =get_setup(args.setup,args.device)

    temperatures = [float(n) for n in args.temperatures.split(',')]

    start = timeit.default_timer()
    theta_ens, ladderAcceptanceRate, swapAcceptanceRate =learning(setup.logposterior, setup.param_count, setup.device, args.numiter, args.burnin, args.thinning, temperatures, args.maintempindex, args.baseMHproposalNoise, args.temperatureNoiseReductionFactor, args.std_init, args.optimize)
    stop = timeit.default_timer()
    execution_time = stop - start

    xpname = setup.experiment_name + '/PTMCMC'
    mlflow.set_experiment(xpname)

    with mlflow.start_run():

        theta=theta_ens

        log_exp_params(setup.param_count, len(theta_ens), ladderAcceptanceRate, swapAcceptanceRate, args.numiter, args.burnin, args.thinning, temperatures, args.maintempindex, args.baseMHproposalNoise, args.temperatureNoiseReductionFactor, args.std_init, args.optimize, args.device)
        theta = torch.stack(theta_ens)

        log_exp_metrics(setup.evaluate_metrics,theta,execution_time,'cpu')

        if setup.plot:
            draw_experiment(setup.makePlot, theta, 'cpu')
        #
        save_params_ens(theta)



