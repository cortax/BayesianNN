from torch import nn
import argparse
import mlflow
import timeit

from Inference.GeNVI_method import GeNVariationalInference, GeNetEns
from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model


## command line example
# python -m Experiments.GeNVI --setup=foong --verbose=True --max_iter=10

def GeNVI_learning(objective_fn,
                   ensemble_size, lat_dim, layerwidth, param_count, activation, init_w, init_b,
                   kNNE, n_samples_NNE, n_samples_KDE, n_samples_ED, n_samples_LP,
                   max_iter, learning_rate, min_lr, patience, lr_decay,
                   device=None, verbose=False):
	GeN = GeNetEns(ensemble_size, lat_dim, layerwidth, param_count,
	               activation, init_w, init_b, device)

	optimizer = GeNVariationalInference(objective_fn,
	                                    kNNE, n_samples_NNE, n_samples_KDE, n_samples_ED, n_samples_LP,
	                                    max_iter, learning_rate, min_lr, patience, lr_decay,
	                                    device, verbose)
	the_epoch, the_scores = optimizer.run(GeN)
	log_scores = [optimizer.score_elbo, optimizer.score_entropy, optimizer.score_logposterior]
	return GeN, the_epoch, the_scores, log_scores


def log_GeNVI_experiment(setup, the_epoch, the_scores, log_scores,
                         ensemble_size, lat_dim, layerwidth, param_count, init_w,
                         kNNE, n_samples_NNE, n_samples_KDE, n_samples_ED, n_samples_LP,
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device):


	mlflow.set_tag('device', device)
	mlflow.set_tag('sigma noise', setup.sigma_noise)
	mlflow.set_tag('dimensions', setup.param_count)

	if kNNE == 0:
		entropy_mthd = 'KDE'
	else:
		entropy_mthd = str(kNNE) + 'NNE'

	mlflow.set_tag('entropy', entropy_mthd)

	mlflow.log_param('ensemble_size', ensemble_size)
	mlflow.log_param('lat_dim', lat_dim)
	mlflow.log_param('layerwidth', layerwidth)
	mlflow.log_param('init_w', init_w)

	mlflow.log_param('n_samples_NNE', n_samples_NNE)
	mlflow.log_param('n_samples_KDE', n_samples_KDE)
	mlflow.log_param('n_samples_ED', n_samples_ED)
	mlflow.log_param('n_samples_LP', n_samples_LP)

	mlflow.log_param('learning_rate', learning_rate)
	mlflow.log_param('patience', patience)
	mlflow.log_param('lr_decay', lr_decay)
	mlflow.log_param('max_iter', max_iter)
	mlflow.log_param('min_lr', min_lr)

	mlflow.log_metric('The epoch', the_epoch)

	mlflow.log_metric("The elbo", float(the_scores[0]))
	mlflow.log_metric("The entropy", float(the_scores[1]))
	mlflow.log_metric("The logposterior", float(the_scores[2]))

	for t in range(len(log_scores[0])):
		mlflow.log_metric("elbo", float(log_scores[0][t]), step=t)
		mlflow.log_metric("entropy", float(log_scores[1][t]), step=t)
		mlflow.log_metric("logposterior", float(log_scores[2][t]), step=t)




parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default=None,
                    help="data setup on which run the method")
parser.add_argument("--ensemble_size", type=int, default=1,
                    help="number of hypernets to train in the ensemble")
parser.add_argument("--lat_dim", type=int, default=5,
                    help="number of latent dimensions of each hypernet")
parser.add_argument("--layerwidth", type=int, default=50,
                    help="layerwidth of each hypernet")
parser.add_argument("--init_w", type=float, default=0.2,
                    help="std for weight initialization of output layers")
#    parser.add_argument("--init_b", type=float, default=0.000001,
#                        help="std for bias initialization of output layers")
parser.add_argument("--EntropyE", type=int, default=0,
                    help="k≥1 Nearest Neighbor Estimate, 0 is for KDE")
parser.add_argument("--n_samples_NNE", type=int, default=500,
                    help="number of samples for NNE")
parser.add_argument("--n_samples_KDE", type=int, default=1000,
                    help="number of samples for KDE")
parser.add_argument("--n_samples_ED", type=int, default=50,
                    help="number of samples for MC estimation of differential entropy")
parser.add_argument("--n_samples_LP", type=int, default=100,
                    help="number of samples for MC estimation of expected logposterior")
parser.add_argument("--max_iter", type=int, default=10000,
                    help="maximum number of learning iterations")
parser.add_argument("--learning_rate", type=float, default=0.08,
                    help="initial learning rate of the optimizer")
parser.add_argument("--min_lr", type=float, default=0.00000001,
                    help="minimum learning rate triggering the end of the optimization")
parser.add_argument("--patience", type=int, default=400,
                    help="scheduler patience")
parser.add_argument("--lr_decay", type=float, default=.5,
                    help="scheduler multiplicative factor decreasing learning rate when patience reached")
parser.add_argument("--device", type=str, default=None,
                    help="force device to be used")
parser.add_argument("--verbose", type=bool, default=False,
                    help="force device to be used")

if __name__ == "__main__":
	# python -m Experiments.boston.GeNVI --max_iter=10 --verbose=True --device='cpu'

	args = parser.parse_args()
	print(args)

	setup = get_setup(args.setup, args.device)

	activation = nn.ReLU()
	init_b = .001

	start = timeit.default_timer()
	GeN, the_epoch, the_scores, log_scores = GeNVI_learning(setup.logposterior,
	                                                        args.ensemble_size, args.lat_dim, args.layerwidth,
	                                                        setup.param_count,
	                                                        activation, args.init_w, init_b,
	                                                        args.EntropyE, args.n_samples_NNE, args.n_samples_KDE,
	                                                        args.n_samples_ED, args.n_samples_LP,
	                                                        args.max_iter, args.learning_rate, args.min_lr,
	                                                        args.patience, args.lr_decay,
	                                                        args.device, args.verbose)
	stop = timeit.default_timer()
	execution_time = stop - start

	xpname = setup.experiment_name + '/GeNVI'
	mlflow.set_experiment(xpname)
	expdata = mlflow.get_experiment_by_name(xpname)

	with mlflow.start_run(experiment_id=expdata.experiment_id):
		log_GeNVI_experiment(setup, the_epoch, the_scores, log_scores,
		                     args.ensemble_size, args.lat_dim, args.layerwidth, setup.param_count, args.init_w,
		                     args.EntropyE, args.n_samples_NNE, args.n_samples_KDE, args.n_samples_ED,
		                     args.n_samples_LP,
		                     args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay,
		                     args.device)

		theta = GeN(1000).detach().cpu()
		log_exp_metrics(setup.evaluate_metrics, theta, execution_time, 'cpu')

		save_model(GeN)

		if setup.plot:
			theta_ens = GeN(1000).detach().cpu()
			draw_experiment(setup.makePlot, theta_ens, 'cpu')
