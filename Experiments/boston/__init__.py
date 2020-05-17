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
sigma_noise = 1.5
seed = 1
sigma_prior=.4

class Setup(AbstractRegressionSetup): 
    def __init__(self,  device, seed= seed):
        super(Setup, self).__init__()

        self.experiment_name = experiment_name
        self.sigma_noise = sigma_noise
        self.sigma_prior=sigma_prior

        
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
    
    def projection(self,theta,n_samples,ratio_ood):
        #compute size of both samples
        n_id=int((1.-ratio_ood)*n_samples)
        n_ood=int(ratio_ood*n_samples)
        
        #batch sample from train
        index=torch.randperm(self._X_train.shape[0])
        X_id=self._X_train[index][0:n_id]
        
        #sample from ood distribution 
        #here uniform [train.min()-2,train.max()+2] in each coordinate
        m,_=self._X_train.min(dim=0)
        M,_=self._X_train.max(dim=0)
        X_ood=torch.Tensor(n_ood,input_dim)
        for i in range(input_dim):
            X_ood[:,i].uniform_(m[i]-2,M[i]+2)
        # here is using a normal instead   
        #ood_samples=torch.Tensor(n_ood,input_dim).normal_(0.,3.).to(self.device)
        X=torch.cat([X_id, X_ood.to(self.device)])
        theta_proj=self._model(X, theta).squeeze(2)
        return theta_proj
    
    def prediction(self,X,theta):
        y_pred=self._normalized_prediction(X, theta, self.device).squeeze(2)
        return y_pred
    
    def train_data(self):
        return self._X_train, self._y_train

    def test_data(self):
        return self._X_test, self._y_test
        