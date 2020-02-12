import numpy as np
import torch
from torch import nn

from Experiments import AbstractRegressionSetup
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from Models import get_mlp
from Tools import logmvn01pdf, log_norm, NormalLogLikelihood
from Preprocessing import fitStandardScalerNormalization, normalize
from Metrics import avgNLL

#exp_path="Experiments/boston/"
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

    def _split_holdout_data(self):
        X_tv, self._X_test, y_tv, self._y_test = train_test_split(self._X, self._y, test_size=0.20, random_state=seed)
        self._X_train, self._X_validation, self._y_train, self._y_validation = train_test_split(X_tv, y_tv, test_size=0.25, random_state=seed)

    def _normalize_data(self):
        self._scaler_X, self._scaler_y = fitStandardScalerNormalization(self._X_train, self._y_train)
        self._X_train, self._y_train = normalize(self._X_train, self._y_train, self._scaler_X, self._scaler_y)
        self._X_validation, self._y_validation = normalize(self._X_validation, self._y_validation, self._scaler_X, self._scaler_y)
        self._X_test, self._y_test = normalize(self._X_test, self._y_test, self._scaler_X, self._scaler_y)

    def _flip_data_to_torch(self):
        self._X = torch.tensor(self._X, device=self.device).float()
        self._y = torch.tensor(self._y, device=self.device).float()
        self._X_train = torch.tensor(self._X_train, device=self.device).float()
        self._y_train = torch.tensor(self._y_train, device=self.device).float()
        self._X_validation = torch.tensor(self._X_validation, device=self.device).float()
        self._y_validation = torch.tensor(self._y_validation, device=self.device).float()
        self._X_test = torch.tensor(self._X_test, device=self.device).float()
        self._y_test = torch.tensor(self._y_test, device=self.device).float()

    def _logprior(self, theta):
        return logmvn01pdf(theta)
    
        # il faudra des méthodes normalize/inverse, car il la normalization est hard-coder
    def _normalized_prediction(self, X, theta):
        """Predict raw inverse normalized values for M models on N data points of D-dimensions
        Arguments:
            X {[tensor]} -- Tensor of size NxD 
            theta {[type]} -- Tensor[M,:] of models
        
        Returns:
            [tensor] -- MxNx1 tensor of predictions
        """
        y_pred = self._model(X, theta)
        if hasattr(self, '_scaler_y'):
            y_pred = y_pred * torch.tensor(self._scaler_y.scale_, device=self.device) + torch.tensor(self._scaler_y.mean_, device=self.device)
        return y_pred

    def _loglikelihood(self, theta, X, y):
        """
        parameters:
            theta (Tensor): M x param_count (models)
            X (Tensor): N x input_dim
            y (Tensor): N x 1
        output:
            LL (Tensor): M x N (models x data)
        """
        y_pred = self._normalized_prediction(X, theta) # MxNx1 tensor
        return NormalLogLikelihood(y_pred, y, sigma_noise) # MxN

    def logposterior(self, theta):
        return self._logprior(theta) + self._loglikelihood(theta, self._X_train, self._y_train).sum(dim=1)

    # Il faudra ajouter les métrique in-between pour foong (spécifique donc ne pas remonter cette méthode)
    # def evaluate_metrics(self, theta):
    #     theta = theta.to(self.device)
    #     avgNLL_train = avgNLL(self._loglikelihood, theta, self._X_train, self._y_train)
    #     avgNLL_validation = avgNLL(self._loglikelihood, theta, self._X_validation, self._y_validation)
    #     avgNLL_test = avgNLL(self._loglikelihood, theta, self._X_test, self._y_test)
    #
    #     return avgNLL_train, avgNLL_validation, avgNLL_test

    def evaluate_metrics(self, theta):
        theta = theta.to(self.device)
        nLPP_train = nLPP(self._loglikelihood, theta, self._X_train, self._y_train)
        nLPP_validation = nLPP(self._loglikelihood, theta, self._X_validation, self._y_validation)
        nLPP_test = nLPP(self._loglikelihood, theta, self._X_test, self._y_test)

        RSE_train = RSE(self._normalized_prediction, theta, self._X_train, self._y_train)
        RSE_validation = RSE(self._normalized_prediction, theta, self._X_validation, self._y_validation)
        RSE_test = RSE(self._normalized_prediction, theta, self._X_test, self._y_test)
        return nLPP_train, nLPP_validation, nLPP_test, RSE_train, RSE_validation, RSE_test
        