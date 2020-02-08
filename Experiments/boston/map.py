import torch
from torch import nn
import mlflow
import mlflow.pytorch
import tempfile
import argparse

from Inference.map_method import main, getParser
import Experiments.boston.setup as data

if __name__== "__main__":
    parser = getParser()
    args = parser.parse_args()
    print(args)

    if args.device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    main(data.get_data, data.get_model, data.sigma_noise, data.experiment_name, data.nb_split, args.ensemble_size,
        args.init_std, args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, device, args.verbose)

