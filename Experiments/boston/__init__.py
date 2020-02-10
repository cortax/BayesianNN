import numpy as np
import torch
from torch import nn

#from Experiments import AbstractSetup
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from Models import get_mlp
from Tools import logmvn01pdf, log_norm
from Preprocessing import fitStandardScalerNormalization, normalize

#exp_path="Experiments/boston/"
#experiment_name='Boston'

input_dim = 13
nblayers = 1
activation = nn.ReLU()
layerwidth = 50
sigma_noise = 1.0
seed = 42

class Setup(): 
    def __init__(self, device):
        self.device = device
        self.param_count, self._model = get_mlp(input_dim, layerwidth, nblayers, activation)

        self._preparare_data()
        self._split_holdout_data()
        self._normalize_data()
        self._flip_data_to_torch()

    def _preparare_data(self):
        self._X, _y = load_boston(return_X_y=True)
        self._y = np.expand_dims(_y, axis=1)

    def _split_holdout_data(self):
        X_tv, self._X_test, y_tv, self._y_test = train_test_split(self._X, self._y, test_size=0.20, random_state=seed)
        self._X_train, self._X_validation, self._y_train, self._y_validation = train_test_split(X_tv, y_tv, test_size=0.25, random_state=seed)

    def _normalize_data(self):
        self._scaler_X, self._scaler_y = fitStandardScalerNormalization(self._X_train, self._y_train)
        self._X_train, self._y_train = normalize(self._X_train, self._y_train, self._scaler_X, self._scaler_y)
        self._X_validation, self._y_validation = normalize(self._X_validation, self._y_validation, self._scaler_X, self._scaler_y)
        self._X_test, self._y_test = normalize(self._X_test, self._y_test, self._scaler_X, self._scaler_y)

    def _flip_data_to_torch(self):
        self._X = torch.tensor(self._X).float()
        self._y = torch.tensor(self._y).float()
        self._X_train = torch.tensor(self._X_train).float()
        self._y_train = torch.tensor(self._y_train).float()
        self._X_validation = torch.tensor(self._X_validation).float()
        self._y_validation = torch.tensor(self._y_validation).float()
        self._X_test = torch.tensor(self._X_test).float()
        self._y_test = torch.tensor(self._y_test).float()

    def _logprior(self, theta):
        return logmvn01pdf(theta)
    
    def _loglikelihood(self, theta, X, y):
        """
        Evaluation of log normal N(y|y_pred,sigma_noise^2*I)

        Parameters:
        x (Tensor): Data tensor of size NxD

        Returns:
        logproba (Tensor): N-dimensional vector of log probabilities
        """
        y_pred = self._model(X, theta)
        if hasattr(self, '_scaler_y'):
            y_pred = y_pred*torch.tensor(self._scaler_y.scale_) + torch.tensor(self._scaler_y.mean_)
        return log_norm(y_pred, y, sigma_noise)

    def logposterior(self, theta):
        return self._logprior(theta) + torch.sum(self._loglikelihood(theta, self._X_train, self._y_train), dim=1).squeeze() 

    def _avgNLL(self, theta, X, y):
        L = self._loglikelihood(theta, X, y)
        n_theta = torch.as_tensor(float(theta.shape[0]), device=self.device)
        log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(n_theta)
        return torch.mean(-log_posterior_predictive)

        