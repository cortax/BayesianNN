import torch
from torch import nn
import tempfile
import argparse

from Inference.GeNVI_method import main, getParser
import Experiments.boston.setup as data


if __name__== "__main__":

    parser = getParser()
    args = parser.parse_args()
    print(args)
    
    if args.device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
    
    main(data.get_data,data.get_model,data.sigma_noise,data.experiment_name, args.ensemble_size,args.lat_dim,args.layerwidth, args.init_w, args.NNE, args.n_samples_NNE, args.n_samples_KDE, args.n_samples_ED, args.n_samples_LP, args.max_iter, args.learning_rate, args.min_lr, args.patience, args.lr_decay, device, args.verbose,args.show_metrics)

