import torch
from torch import nn
import tempfile
import argparse

from Inference.GeNVI_method import GeNVI, GeNetEns
from Experiments.boston import Setup

## command line example
# python -m Experiments.boston.GeNVI --verbose=True --max_iter=1000 --show_metrics=True

parser = argparse.ArgumentParser()
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
parser.add_argument("--NNE", type=int, default=0,
                    help="k≥1 Nearest Neighbor Estimate, 0 is for KDE")
parser.add_argument("--n_samples_NNE", type=int, default=500,
                    help="number of samples for NNE")
parser.add_argument("--n_samples_KDE", type=int, default=500,
                    help="number of samples for KDE")
parser.add_argument("--n_samples_ED", type=int, default=50,
                    help="number of samples for MC estimation of differential entropy")
parser.add_argument("--n_samples_LP", type=int, default=100,
                    help="number of samples for MC estimation of expected logposterior")
parser.add_argument("--max_iter", type=int, default=100000,
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
parser.add_argument("--show_metrics", type=bool, default=False,
                    help="log metrics during training")




if __name__== "__main__":
    #python -m Experiments.boston.GeNVI --max_iter=10 --verbose=True --device='cpu'

    args = parser.parse_args()
    print(args)
    
    if args.device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    boston = Setup(args.device)

    GeN=GeNetEns(args.ensemble_size, args.lat_dim, args.layerwidth, boston.param_count,
                 activation=nn.ReLU(),init_w=args.init_w,init_b=.001, device=device)

    GeNVI(boston.logposterior, GeN, args.NNE, args.n_samples_NNE, args.n_samples_KDE, args.n_samples_ED, args.n_samples_LP, args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, device, args.verbose)

    Epoch, ELBO, ED, LP = GeN._get_best_model()
    best='Epoch {}, Training Loss: {}, Entropy: {}, -LogPosterior {}'.format(Epoch, ELBO, ED, LP)
    print(best)

