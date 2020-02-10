import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from Experiments import AbstractRegressionSetup
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from Models import get_mlp
from Tools import logmvn01pdf, log_norm, NormalLogLikelihood
from Preprocessing import fitStandardScalerNormalization, normalize
from Metrics import avgNLL

experiment_name = 'Foong'
data_path='Experiments/foong/data/'
#exp_path="Experiments/foong/"

input_dim = 1
nblayers = 1
activation = nn.Tanh()
layerwidth = 50
sigma_noise = 0.1
seed = 42

class Setup(AbstractRegressionSetup):  
    def __init__(self, device):
        super(Setup).__init__()
        self.experiment_name = experiment_name
        self.sigma_noise = sigma_noise

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

    # il faudra des méthodes normalize/inverse, car il la normalization est hard-coder
    def _normalized_prediction(self, X, theta):
        """Predict raw inverse normalized values for M models on N data points of D-dimensions
        Arguments:
            X {[tensor]} -- Tensor of size NxD 
            theta {[type]} -- Tensor[M,:] of models
        
        Returns:
            [tensor] -- MxNx1 tensor of predictions
        """
        assert type(theta) is torch.Tensor
        y_pred = self._model(X, theta)
        if hasattr(self, '_scaler_y'):
            y_pred = y_pred * torch.tensor(self._scaler_y.scale_, device=self.device) + torch.tensor(self._scaler_y.mean_, device=self.device)
        return y_pred

    def _loglikelihood(self, theta, X, y, raw=False):
        y_pred = self._normalized_prediction(X, theta) # MxNx1 tensor
        return NormalLogLikelihood(y_pred, y, sigma_noise, raw)

    def logposterior(self, theta):
        return self._logprior(theta) + self._loglikelihood(theta, self._X_train, self._y_train)

    # Il faudra ajouter les métrique in-between pour foong (spécifique donc ne pas remonter cette méthode)
    def evaluate_metrics(self, theta):
        theta = theta.to(self.device)
        avgNLL_train = avgNLL(self._loglikelihood, theta, self._X_train, self._y_train)
        avgNLL_validation = avgNLL(self._loglikelihood, theta, self._X_validation, self._y_validation)
        avgNLL_test = avgNLL(self._loglikelihood, theta, self._X_test, self._y_test)
        return avgNLL_train, avgNLL_validation, avgNLL_test

    def makeValidationPlot(self, theta):
        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2) 
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Validation set')
        plt.scatter(self._X_validation.cpu(), self._y_validation.cpu())
        for i in range(theta.shape[0]):
            y_pred = self._normalized_prediction(x_lin, theta[i,:].unsqueeze(0)) 
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0, linewidth=1.0, color='black')
            plt.fill_between(x_lin.detach().cpu().numpy().squeeze(), y_pred.squeeze(0).detach().cpu().numpy().squeeze()-3*self.sigma_noise, y_pred.squeeze(0).detach().cpu().numpy().squeeze()+3*self.sigma_noise, alpha=0.5, color='lightblue')
        return plt
        

    

        
