import torch
from torch import nn
import matplotlib.pyplot as plt

from Experiments import AbstractRegressionSetup

from Models import get_mlp

import numpy as np

from Tools import logmvn01pdf, NormalLogLikelihood

experiment_name = 'Foong'
data_path='Experiments/foong/data/'

input_dim = 1
nblayers = 1
activation = nn.Tanh()
layerwidth = 50
sigma_noise = 0.1
seed = 42

class Setup(AbstractRegressionSetup):  
    def __init__(self, device, layerwidth=layerwidth, nblayers=nblayers):
        super(Setup).__init__()
        self.experiment_name = experiment_name
        self.sigma_noise = sigma_noise
        
        self.plot = True

        self.device = device
        self.param_count, self._model = get_mlp(input_dim, layerwidth, nblayers, activation)
        self._preparare_data()
        

    def _preparare_data(self):
        train = torch.load(data_path + 'foong_train_sparse.pt')
        valid = torch.load(data_path + 'foong_validation_sparse.pt')
        test = torch.load(data_path + 'foong_test.pt')
        
        self._X_train, self._y_train = train[0].to(self.device), train[1].unsqueeze(-1).to(self.device)
        self._X_validation, self._y_validation = valid[0].to(self.device), valid[1].unsqueeze(-1).to(self.device)
        self._X_test, self._y_test = test[0].to(self.device), test[1].unsqueeze(-1).to(self.device)
        self.n_train_samples=self._X_train.shape[0]


    def makePlot(self, theta, device):
        def get_linewidth(linewidth, axis):
            fig = axis.get_figure()
            ppi = 72  # matplolib points per inches
            length = fig.bbox_inches.height * axis.get_position().height
            value_range = np.diff(axis.get_ylim())[0]
            return linewidth * ppi * length / value_range
        nb_samples_plot=theta.shape[0]
        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1)
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        plt.xlim(-2, 2) 
        plt.ylim(-4, 6)
        plt.grid(True, which='major', linewidth=0.5)
#        plt.title('Validation set')


        my_lw=get_linewidth(0.2,ax)
        alpha = (.9 / torch.tensor(float(nb_samples_plot)).sqrt()).clamp(0.05, 1.)
        theta=theta.detach().to(device)
        for i in range(theta.shape[0]):
            y_pred = self._normalized_prediction(x_lin, theta[i,:].unsqueeze(0), device)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.8*alpha, linewidth=my_lw,
                     color='springgreen', zorder=2)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=alpha, linewidth=1.0, color='green',zorder=3)
         #   plt.fill_between(x_lin.detach().cpu().numpy().squeeze(), y_pred.squeeze(0).detach().cpu().numpy().squeeze()-3*self.sigma_noise, y_pred.squeeze(0).detach().cpu().numpy().squeeze()+3*self.sigma_noise, alpha=0.5, color='lightblue')
        plt.scatter(self._X_train.cpu(), self._y_train.cpu(), marker='.',color='black',zorder=4)
        return fig

    def loglikelihood(self, theta):
        ll=torch.sum(self._loglikelihood(theta, self._X_train, self._y_train, self.device),dim=1)
        return ll
    
    def loss(self,theta, R):
        y_pred = self._normalized_prediction(self._X_train, theta, self.device)  # MxNx1 tensor
        assert y_pred.shape[1] == self._y_train.shape[0]
        assert y_pred.shape[2] == self._y_train.shape[1]
        assert self._y_train.shape[1] == 1
        B = y_pred.shape[0]
        S = y_pred.shape[1]
        d = torch.tanh(R*(y_pred.view(B, S, 1) - self._y_train.view(1, S, 1)) ** 2)
        return d.mean(1)

    def logprior(self, theta):
        return  self._logprior(theta)
    
    def projection(self,theta,k):
        X=torch.Tensor(k,input_dim).uniform_(-2.,2.).to(self.device)
        theta_proj=self._normalized_prediction(X, theta, self.device).squeeze(2)
        #theta_proj=self._normalized_prediction(self._X_train, theta, self.device).squeeze(2)
        return theta_proj
    
    def prediction(self,X,theta):
        y_pred=self._normalized_prediction(X, theta, self.device).squeeze(2)
        #theta_proj=self._normalized_prediction(self._X_train, theta, self.device).squeeze(2)
        return y_pred
    
    
    def train_data(self):
        return self._X_train, self._y_train
        


        
