import numpy as np
import torch
from torch import nn

from Experiments import AbstractRegressionSetup
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from Models import get_mlp
from Preprocessing import fitStandardScalerNormalization, normalize

experiment_name='Boston'

input_dim = 13
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
        self._X, _y = load_boston(return_X_y=True)
        self._y = np.expand_dims(_y, axis=1)
        
    def loglikelihood(self, theta):
        ll=torch.sum(self._loglikelihood(theta, self._X_train, self._y_train, self.device),dim=1)
        return ll

    def logprior(self, theta):
        return  self._logprior(theta)
    
    def projection(self,theta,k):
        X=torch.Tensor(k,input_dim).normal_(0.,1.).to(self.device)
        theta_proj=self._normalized_prediction(X, theta, self.device).squeeze(2)
        return theta_proj


        