import sys
import os
from os.path import dirname
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dirname(dirname(dirname(dirname(cwd)))) )

from Inference import BBVI 
import torch


def train_model(layer_width, nb_layers, activation):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    data = torch.load('data/foong_data.pt')
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = y_data.unsqueeze(-1)

    optimizer = torch.optim.Adam
    optimizer_params = {'lr': 0.1}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {'patience': 5, 'factor': 0.8}

    Net = BBVI.VariationalNetwork(input_size=1, output_size=1, layer_width=layer_width, nb_layers=nb_layers, activation=activation, device=device)
    Net.make_deterministic_rhos()
    Net.requires_grad_rhos(False)

    voptimizer = BBVI.VariationalOptimizer(model=Net, sigma_noise=0.1, optimizer=optimizer, optimizer_params=optimizer_params, scheduler=scheduler, scheduler_params=scheduler_params, min_lr=0.00001)
    Net = voptimizer.run((x_data,y_data), n_epoch=int(250+2*i), n_iter=250, seed=seed, n_ELBO_samples=1, plot=True)

    return Net

if __name__ == "__main__":
    activation = torch.tanh
    Net = train_model(10, 2, activation)

    

