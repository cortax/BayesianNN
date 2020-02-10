import numpy as np
import torch
from torch import nn

from Experiments import AbstractRegressionSetup
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from Models import get_mlp
from Tools import logmvn01pdf, log_norm, NormalLogLikelihood
from Preprocessing import fitStandardScalerNormalization, normalize

data_path='Experiments/foong/data/'
#exp_path="Experiments/foong/"
experiment_name='foong'

input_dim = 1
nblayers = 1
activation = nn.Tanh()
layerwidth = 50
sigma_noise = 0.1
seed = 42

class Setup(AbstractRegressionSetup):  
    def __init__(self, device):
        super(Setup).__init__()
        self.device = device
        self.param_count, self._model = get_mlp(input_dim, layerwidth, nblayers, activation)
        self._preparare_data()

    def _preparare_data(self):
        train = torch.load(data_path + 'foong_train.pt')
        valid = torch.load(data_path + 'foong_validation.pt')
        test = torch.load(data_path + 'foong_test.pt')
        
        self._X_train, self._y_train = train[0].to(self.device), train[1].unsqueeze(-1).to(self.device)
        self._X_validation, self._y_validation = valid[0].to(self.device), valid[1].unsqueeze(-1).to(self.device)
        self._X_test, self._y_test = test[0].to(self.device), test[1].unsqueeze(-1).to(self.device)

    def _logprior(self, theta):
        return logmvn01pdf(theta)

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
        y_pred = self._normalized_prediction(X, theta) # MxNx1 tensor
        return NormalLogLikelihood(y_pred, y, sigma_noise)

    def _logposterior(self, theta, X, y):
        return self._logprior(theta) + self._loglikelihood(theta, X, y)

    def logposterior(self, theta):
        return self._logprior(theta) + self._loglikelihood(theta, self._X_train, self._y_train)

    # def _avgNLL(self, theta, X, y):
    #     L = self._loglikelihood(theta, X, y)
    #     n_theta = torch.as_tensor(float(theta.shape[0]), device=self.device)
    #     log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(n_theta)
    #     return torch.mean(-log_posterior_predictive)

        
