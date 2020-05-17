import numpy as np
import torch
from torch import nn

from Experiments import AbstractRegressionSetup

from Models import get_mlp

experiment_name='Kin8nm'

input_dim = 8
nblayers = 1
activation = nn.ReLU()
layerwidth = 50
sigma_noise = 1.0  #yarin gal 0.07
seed = 37

class Setup(AbstractRegressionSetup): 
    def __init__(self, device):
        self.experiment_name = experiment_name
        self.sigma_noise = sigma_noise

        self.plot = False

        self.device = device
        self.param_count, self._model = get_mlp(input_dim, layerwidth, nblayers, activation)

        self._preparare_data()
        self._split_holdout_data()
        self._normalize_data()
        self._flip_data_to_torch()

    def _preparare_data(self):
        self._X, _y = torch.load ('Experiments/kin8nm/data.pt')
        self._y = np.expand_dims(_y, axis=1)




        