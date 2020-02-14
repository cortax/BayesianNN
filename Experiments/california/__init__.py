import numpy as np
import torch
from torch import nn

from Experiments import AbstractRegressionSetup
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from Models import get_mlp
from Preprocessing import fitStandardScalerNormalization, normalize

experiment_name='California'

input_dim = 8
nblayers = 1
activation = nn.ReLU()
layerwidth = 50
sigma_noise = 1.0
seed = 42

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
        self._X, _y = fetch_california_housing(return_X_y=True)
        self._y = np.expand_dims(_y, axis=1)




        