import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


from Prediction.mlp import *


import mlflow
import tempfile


experiment_name='Foong/L1W50'

data_path='Experiments/foong/data/'

input_dim=1
nblayers = 1
activation=nn.Tanh()
layerwidth = 50



sigma_noise=.1



def get_my_mlp():
    return get_mlp(input_dim,layerwidth,nblayers,activation)


def get_data(device):
    train = torch.load(data_path+'foong_train.pt')
    test = torch.load(data_path+'foong_test.pt')
    ib_test = torch.load(data_path+'foong_test_in_between.pt')
    valid =torch.load(data_path+'foong_validation.pt')
    x_train,y_train=train[0].to(device),train[1].unsqueeze(-1).to(device)
    x_test,y_test=test[0].to(device),test[1].unsqueeze(-1).to(device)
    x_ib_test,y_ib_test=ib_test[0].to(device),ib_test[1].unsqueeze(-1).to(device)
    x_valid,y_valid=valid[0].to(device),valid[1].unsqueeze(-1).to(device)
    return x_train, y_train, x_test, y_test, x_ib_test, y_ib_test, x_valid, y_valid, lambda x: x


def get_linewidth(linewidth, axis):
    fig = axis.get_figure()
    ppi=72 #matplolib points per inches
    length = fig.bbox_inches.height * axis.get_position().height
    value_range = np.diff(axis.get_ylim())[0]
    return linewidth*ppi*length/value_range

def plot_test(x_train, y_train, x_test,y_test,theta, model):
    tempdir = tempfile.TemporaryDirectory()
    
    
    ensemble_sz=theta.shape[0]
    nb_samples_plot=theta.shape[1]
    
    x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1)
    theta=theta.cpu()
    
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    plt.xlim(-2, 2) 
    plt.ylim(-4, 4)
    plt.grid(True, which='major', linewidth=0.5)
    plt.title('Test set')

    plt.scatter(x_test.cpu(), y_test.cpu(),alpha=0.05,color='dimgrey',zorder=0)
    plt.scatter(x_train.cpu(), y_train.cpu(),color='black',zorder=1)    

    plt_linewidth=get_linewidth(2*sigma_noise,ax)
    alpha=(.9/torch.tensor(float(ensemble_sz*nb_samples_plot)).sqrt()).clamp(0.01,1.)
    for c in range(ensemble_sz):
        for i in range(nb_samples_plot):
            y_pred = model(x_lin.cpu(),theta[c,i].unsqueeze(0))
            plt.plot(x_lin, y_pred.squeeze(0), alpha=alpha, linewidth=plt_linewidth, color='C'+str(c+2),zorder=2) 

    fig.savefig(tempdir.name+'/test.png', dpi=5*fig.dpi)
    mlflow.log_artifact(tempdir.name+'/test.png')
    plt.close()


    