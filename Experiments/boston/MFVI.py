import torch
from torch import nn
import mlflow
import mlflow.pytorch
import tempfile
import argparse
import numpy as np

from Inference.MFVI_method import main, parser


import Experiments.boston.setup as data



if __name__ == "__main__":    
    args = parser.parse_args()

    print(args)

if args.seed is None:
    seed = np.random.randint(0, 2 ** 31)
else:
    seed = args.seed

if args.device is None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = args.device


main(data.get_data,data.get_model,data.sigma_noise,data.experiment_name, data.nb_split, args.ensemble_size, args.max_iter, args.learning_rate, args.min_lr,  args.n_samples_ED, args.n_samples_LP,  args.patience, args.lr_decay, args.init_std, args.optimize, seed, device, args.verbose,args.show_metrics)
