import math
import torch
from torch import nn

from NeuralNetwork.mlp import *
from Tools import log_norm

from sklearn.preprocessing import StandardScaler

exp_path="Experiments/boston/"

experiment_name='Boston'

input_dim = 13
nblayers = 1
activation = nn.ReLU()
layerwidth = 50
sigma_noise = 1.0

param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)


def normalize(X_train, y_train, X_test, y_test, device):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = torch.as_tensor(
        scaler_X.fit_transform(X_train)).float().to(device)
    y_train_un = y_train.clone().float().to(device)
    y_train = torch.as_tensor(
        scaler_y.fit_transform(y_train)).float().to(device)

    def inverse_scaler_y(t): return torch.as_tensor(
        scaler_y.inverse_transform(t.cpu())).to(device)

    X_test = torch.as_tensor(scaler_X.transform(X_test)).float().to(device)
    y_test_un = y_test.float().to(device)
    return X_train, y_train, y_train_un, X_test, y_test_un, inverse_scaler_y


def get_data(device):
    splitting_index = 0  # TODO: Faire un train(70)-validation(30)
    X_train = torch.load(
        exp_path+'data/boston_X_train_('+str(splitting_index)+').pt')
    y_train = torch.load(
        exp_path+'data/boston_y_train_('+str(splitting_index)+').pt')
    X_test = torch.load(
        exp_path+'data/boston_X_test_('+str(splitting_index)+').pt')
    y_test = torch.load(
        exp_path+'data/boston_y_test_('+str(splitting_index)+').pt')
    return normalize(X_train, y_train, X_test, y_test, device)


def get_logposterior(device):
    """
    Provides the logposterior function 

    Parameters:
    device: indicates where to computation is done

    Returns:
    function: a function taking Tensors as arguments for evaluation
    """


    def logprior(theta):
        """
        Evaluation of log proba with prior N(0,I_n)

        Parameters:
        x (Tensor): Data tensor of size NxD

        Returns:
        logproba (Tensor): N-dimensional vector of log probabilities
        """
        dim = theta.shape[1]
        S = torch.ones(dim).type_as(theta)
        mu = torch.zeros(dim).type_as(theta)
        n_x = theta.shape[0]

        H = S.view(dim, 1, 1).inverse().view(1, 1, dim)
        d = ((theta-mu.view(1, dim))**2).view(n_x, dim)
        const = 0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
        return -0.5*(H*d).sum(2).squeeze()-const

    X_train, y_train, _, _, _, _ = get_data(device)
    
    def loglikelihood(theta):
        """
        Evaluation of log normal N(y|y_pred,sigma_noise^2*I)

        Parameters:
        x (Tensor): Data tensor of size NxD

        Returns:
        logproba (Tensor): N-dimensional vector of log probabilities
        """
        y_pred = model(X_train, theta)
        L = log_norm(y_pred, y_train, sigma_noise)
        return torch.sum(L, dim=1).squeeze()

    def logposterior(theta):
        return logprior(theta) + loglikelihood(theta)

    return logposterior
