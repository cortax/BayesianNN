import torch
from torch import nn

from Prediction.mlp import *

from sklearn.preprocessing import StandardScaler

exp_path="Experiments/boston/"

experiment_name='Boston'

input_dim = 13
nblayers = 1
activation = nn.ReLU()
layerwidth = 50

sigma_noise = 1.0

#nb_split=5
#predictive net:

def get_model():
    return get_mlp(input_dim,layerwidth,nblayers,activation)

def normalize(X_train, y_train, X_test, y_test,device):

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = torch.as_tensor(scaler_X.fit_transform(X_train)).float().to(device)
    y_train_un=y_train.clone().float().to(device)
    y_train = torch.as_tensor(scaler_y.fit_transform(y_train)).float().to(device)
   
    
    inverse_scaler_y = lambda t: torch.as_tensor(scaler_y.inverse_transform(t.cpu())).to(device)

    X_test = torch.as_tensor(scaler_X.transform(X_test)).float().to(device)
    y_test_un=y_test.float().to(device)
    return X_train, y_train, y_train_un, X_test, y_test_un,  inverse_scaler_y

def get_data(device):
    splitting_index = 0 # TODO: Faire un train(70)-validation(30)
    X_train = torch.load(exp_path+'data/boston_X_train_('+str(splitting_index)+').pt')
    y_train = torch.load(exp_path+'data/boston_y_train_('+str(splitting_index)+').pt')
    X_test = torch.load(exp_path+'data/boston_X_test_('+str(splitting_index)+').pt')
    y_test=torch.load(exp_path+'data/boston_y_test_('+str(splitting_index)+').pt')
    return normalize(X_train, y_train, X_test, y_test,device)


    