import torch
from torch import nn

from Prediction.mlp import *

from Prediction.logposterior import logposterior

from sklearn.preprocessing import StandardScaler

input_dim=13
nb_layers = 1
activation_pn=nn.ReLU()
layerwidth = 50

sigma_noise=1.

def normalize(X_train, y_train, X_test, y_test,device):

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = torch.as_tensor(scaler_X.fit_transform(X_train)).float().to(device)
    y_train = torch.as_tensor(scaler_y.fit_transform(y_train)).float().to(device)

    inverse_scaler_y = lambda t: torch.as_tensor(scaler_y.inverse_transform(t)).to(device)

    X_test = torch.as_tensor(scaler_X.transform(X_test)).float().to(device)
    y_test=y_test.float().to(device)
    return X_train, y_train, X_test,y_test , inverse_scaler_y

def get_data(splitting_index,device):
    X_train = torch.load('Experiments/Boston/data/boston_X_train_('+str(splitting_index)+').pt')
    y_train = torch.load('Experiments/Boston/data/boston_y_train_('+str(splitting_index)+').pt')
    X_test = torch.load('Experiments/Boston/data/boston_X_test_('+str(splitting_index)+').pt')
    y_test=torch.load('Experiments/Boston/data/boston_y_test_('+str(splitting_index)+').pt')
    return normalize(X_train, y_train, X_test, y_test,device)


    
    