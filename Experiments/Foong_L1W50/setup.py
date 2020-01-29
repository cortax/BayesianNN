import torch
from torch import nn
from Tools.NNtools import *
import tempfile
import mlflow
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd
import math


nblayers = 1
activation=nn.Tanh()
sigma_noise = torch.tensor(0.1)
layerwidth = 50
param_count = 2*layerwidth+(nblayers-1)*(layerwidth**2+layerwidth)+layerwidth+1
experiment_name = 'Foong L1/W50'

"""
def mlp(x, theta):
    '''
    Feedforward neural network used as the observation model for the likelihood

    Parameters:
        x (Tensor): Input of the network of size NbExemples X NbDimensions
        theta (Tensor):  M set of parameters of the network NbModels X NbParam

    Returns:
        Predictions (Tensor) with dimensions NbModels X NbExemples X NbDimensions
    '''
    nb_theta = theta.shape[0]
    theta = theta.split([layerwidth, layerwidth, layerwidth, 1], dim=1)
    nb_x = x.shape[0]
    input_x = x.view(nb_x, 1, 1)
    m1 = torch.matmul(theta[0].view(nb_theta, 1, layerwidth, 1), input_x)
    m2 = m1.add(theta[1].reshape(nb_theta, 1, layerwidth, 1))
    m3 = torch.tanh(m2)
    m4 = torch.matmul(theta[2].view(nb_theta, 1, 1,layerwidth), m3)
    m5 = m4.add(theta[3].reshape(nb_theta, 1, 1, 1))
    return m5.squeeze(-1)
"""

def mlp(x,theta,layerwidth=layerwidth,nb_layers=nblayers,activation=activation):
    """
    Feedforward neural network used as the observation model for the likelihood
    

    Parameters:
        x (Tensor): Input of the network of size NbExemples X NbDimensions
        theta (Tensor):  M set of parameters of the network NbModels X NbParam
        layerwidth (Int): Number of hidden units per layer 
        nb_layers (Int): Number of layers
        activation (Module/Function): activation function of the neural network

    Returns:
        Predictions (Tensor) with dimensions NbModels X NbExemples X NbDimensions
    """        

    nb_theta=theta.shape[0]
    nb_x=x.shape[0]
    split_sizes=[layerwidth]*2+[layerwidth**2,layerwidth]*(nb_layers-1)+[layerwidth,1]
    theta=theta.split(split_sizes,dim=1)
    input_x=x.view(nb_x,1,1)
    m=torch.matmul(theta[0].view(nb_theta,1,layerwidth,1),input_x)
    m=m.add(theta[1].reshape(nb_theta,1,layerwidth,1))
    m=activation(m)
    for i in range(nb_layers-1):
        m=torch.matmul(theta[2*i+2].view(-1,1,layerwidth,layerwidth),m)
        m=m.add(theta[2*i+3].reshape(-1,1,layerwidth,1))
        m=activation(m)
    m=torch.matmul(theta[2*(nb_layers-1)+2].view(nb_theta,1,1,layerwidth),m)
    m=m.add(theta[2*(nb_layers-1)+3].reshape(nb_theta,1,1,1))
    return m.squeeze(-1)


def get_training_data(device):
    training_data = torch.load('Experiments/Foong_L1W50/Data/foong_data.pt')
    x_train = training_data[0].to(device)
    y_train = training_data[1].to(device)
    y_train = y_train.unsqueeze(-1)
    return x_train, y_train


def get_validation_data(device):
    validation_data = torch.load('Experiments/Foong_L1W50/Data/foong_data_validation.pt')
    x_validation = validation_data[0].to(device)
    y_validation = validation_data[1].to(device)
    y_validation = y_validation.unsqueeze(-1)
    return x_validation, y_validation


def get_test_data(device):
    test_data = torch.load('Experiments/Foong_L1W50/Data/foong_data_test.pt')
    x_test = test_data[0].to(device)
    y_test = test_data[1].to(device)
    y_test = y_test.unsqueeze(-1)
    return x_test, y_test


def get_test_ib_data(device):
    test_data = torch.load('Experiments/Foong_L1W50/Data/foong_data_test_in_between.pt')
    x_test = test_data[0].to(device)
    y_test = test_data[1].to(device)
    y_test = y_test.unsqueeze(-1)
    return x_test, y_test

"""
   KDE by components, returns a KDE estimation for x based on batched samples y with corresponding kernels H 

    Parameters:
        x (Tensor): Data tensor for KDE evaluation, size n_ed X dim 
        y (Tensor): Data tensor for KDE computation, size nb_comp X n_kde X dim
        H (Tensor): Diagonals of kernel covariance_matrix, size nb_comp X dim 
        std (Tensor): Positive scalar

    Returns:
        logproba (Tensor): Same size as x
    """

def get_KDE_fn(device):
    def KDE(x,y,H):
        dim=x.shape[-1]
        n_ed=x.shape[0]
        n_comp=y.shape[0]
        n_kde=y.shape[1]
        d=((y.view(n_comp,n_kde,1,dim)-x.view(1,1,n_ed,dim))**2)
        H_=H.view(n_comp,dim,1,1).inverse().view(n_comp,1,1,dim)
        const=0.5*H.log().sum(1)+0.5*dim*torch.tensor(2*math.pi).log()
        const=const.view(n_comp,1,1)
        ln=-0.5*(H_*d).sum(3)-const
        N=torch.as_tensor(float(n_comp*n_kde),device=device)
        return (ln.logsumexp(0).logsumexp(0)-torch.log(N)).unsqueeze(-1)
    return KDE

def _log_norm(x, mu, std):
    """
    Evaluation of 1D normal distribution on tensors

    Parameters:
        x (Tensor): Data tensor
        mu (Tensor): Mean tensor of same size as x
        std (Tensor): Positive scalar

    Returns:
        logproba (Tensor): Same size as x
    """
    return -0.5 * torch.log(2*np.pi*std**2) -(0.5 * (1/(std**2))* (x-mu)**2)


def get_logprior_fn(device):
    S = torch.eye(param_count).to(device)
    mu = torch.zeros(param_count).to(device)
    prior = MultivariateNormal(mu, scale_tril=S)

    def logprior(x):
        v = prior.log_prob(x).unsqueeze(-1)
        return v

    return logprior


def get_loglikelihood_fn(device):
    def loglikelihood(theta, x, y, sigma_noise):
        y_pred = mlp(x, theta)
        L = _log_norm(y_pred, y.unsqueeze(0).repeat(theta.shape[0], 1, 1), torch.tensor([sigma_noise], device=device))
        return torch.sum(L, dim=1)
    return loglikelihood


def get_logposterior_fn(device):
    logprior = get_logprior_fn(device)
    loglikelihood = get_loglikelihood_fn(device)
    def logposterior(theta, x, y, sigma_noise):
        return logprior(theta).add(loglikelihood(theta, x, y, sigma_noise))
    return logposterior


#NLPD from Quinonero-Candela and al.
# the average negative log predictive density (NLPD) of the true targets
def get_NLPD_fn(device):
    def NLPD(theta, x, y, sigma_noise):
        y_pred = mlp(x, theta)
        L = _log_norm(y_pred, y, torch.tensor([sigma_noise], device=device))
        n_x = torch.as_tensor(float(x.shape[0]), device=device)
        n_theta = torch.as_tensor(float(theta.shape[0]), device=device)
        log_posterior_predictive = torch.logsumexp(L, 0) - torch.log(n_theta)
        return log_posterior_predictive.mean()
    return NLPD


def get_logposteriorpredictive_fn(device):
    def logposteriorpredictive(ensemble, x, y, sigma_noise):
        complogproba = []
        for theta in ensemble:
            y_pred = mlp(x, theta)
            complogproba.append(-torch.tensor(float(len(ensemble))).log() + _log_norm(y_pred, y, torch.tensor([sigma_noise],device=device)))
        return torch.logsumexp(torch.stack(complogproba), dim=0).sum().unsqueeze(-1)
    return logposteriorpredictive


#input: float, linewidth in data units of the respective y-axis
#       matplotlib axis
#output: float,linewidth in points 
def get_linewidth(linewidth, axis):
    fig = axis.get_figure()
    ppi=72 #matplolib points per inches
    length = fig.bbox_inches.height * axis.get_position().height
    value_range = np.diff(axis.get_ylim())[0]
    return linewidth*ppi*length/value_range

def log_model_evaluation_(ensemble, device):
    with torch.no_grad():
        tempdir = tempfile.TemporaryDirectory(dir='/dev/shm')

        logposterior = get_logposterior_fn(device)
        model = get_parallel_model(device)
        x_train, y_train = get_training_data(device)
        x_validation, y_validation = get_validation_data(device)
        x_test, y_test = get_test_data(device)
        #logtarget = lambda theta: logposterior(theta, model, x_train, y_train, sigma_noise)

        logposteriorpredictive = get_logposteriorpredictive_fn(device)
        train_post = logposteriorpredictive(ensemble, model, x_train, y_train, sigma_noise) / len(y_train)
        mlflow.log_metric("training log posterior predictive", -float(train_post.detach().cpu()))
        val_post = logposteriorpredictive(ensemble, model, x_validation, y_validation, sigma_noise) / len(y_validation)
        mlflow.log_metric("validation log posterior predictive", -float(val_post.detach().cpu()))
        test_post = logposteriorpredictive(ensemble, model, x_test, y_test, sigma_noise) / len(y_test)
        mlflow.log_metric("test log posterior predictive", -float(test_post.detach().cpu()))

        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Training set')
        plt_linewidth=get_linewidth(2*sigma_noise,ax)
        for theta in ensemble:
            set_all_parameters(model, theta)
            y_pred = model(x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.01,
                     linewidth=plt_linewidth, color='black', zorder=80)
            # res = 5
            # for r in range(res):
            #     mass = 1.0 - (r + 1) / res
            #     plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * sigma_noise * ((r + 1) / res),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * sigma_noise * ((r + 1) / res),
            #                      alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_train.cpu(), y_train.cpu(), c='red', zorder=100)
        fig.savefig(tempdir.name + '/training.png',dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/training.png')
        plt.close()

        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Validation set')
        plt_linewidth=get_linewidth(2*sigma_noise,ax)
        for theta in ensemble:
            set_all_parameters(model, theta)
            y_pred = model(x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05,
                     linewidth=plt_linewidth, color='black', zorder=80)
            # res = 5
            # for r in range(res):
            #     mass = 1.0 - (r + 1) / res
            #     plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * sigma_noise * ((r + 1) / res),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * sigma_noise * ((r + 1) / res),
            #                      alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_validation.cpu(), y_validation.cpu(), c='red', zorder=100)
        fig.savefig(tempdir.name + '/validation.png', dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/validation.png')
        plt.close()

        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Test set')
        plt_linewidth=get_linewidth(2*sigma_noise,ax)
        for theta in ensemble:
            set_all_parameters(model, theta)            
            y_pred = model(x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05,
                     linewidth=plt_linewidth, color='black', zorder=80)
            # res = 5
            # for r in range(res):
            #     mass = 1.0 - (r + 1) / res
            #     plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * sigma_noise * ((r + 1) / res),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * sigma_noise * ((r + 1) / res),
            #                      alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_test.cpu(), y_test.cpu(), c='red', zorder=100)
        fig.savefig(tempdir.name + '/test.png', dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/test.png')
        plt.close()

def log_model_evaluation(ensemble, device):
    with torch.no_grad():
        tempdir = tempfile.TemporaryDirectory(dir='/dev/shm')
#        pd.DataFrame(training_loss).to_csv(tempdir.name+'/training_loss.csv', index=False, header=False)
#        mlflow.log_artifact(tempdir.name+'/training_loss.csv')


        
        x_train, y_train = get_training_data(device)
        x_validation, y_validation = get_validation_data(device)
        x_test, y_test = get_test_data(device)
        x_test_ib, y_test_ib = get_test_ib_data(device)


        #ensemble_t = torch.cat(ensemble, dim=0)
        logposteriorpredictive = get_logposteriorpredictive_fn(device)
        train_post = logposteriorpredictive(ensemble, x_train, y_train, sigma_noise)
        mlflow.log_metric("training log posterior predictive", -float(train_post.detach().cpu()))
        val_post = logposteriorpredictive(ensemble, x_validation, y_validation, sigma_noise)
        mlflow.log_metric("validation log posterior predictive", -float(val_post.detach().cpu()))
        test_post = logposteriorpredictive(ensemble, x_test, y_test, sigma_noise)
        mlflow.log_metric("test log posterior predictive", -float(test_post.detach().cpu()))
        test_ib_post = logposteriorpredictive(ensemble, x_test_ib, y_test_ib, sigma_noise)
        mlflow.log_metric("test in-between log posterior predictive", -float(test_ib_post.detach().cpu()))
        
        
        NLPD = get_NLPD_fn(device)
        theta=torch.cat(ensemble,dim=0)
        train_NLPD = NLPD(theta, x_train, y_train, sigma_noise)
        mlflow.log_metric("training NLPD", -float(train_NLPD.detach().cpu()))
        val_NLPD = NLPD(theta, x_validation, y_validation, sigma_noise)
        mlflow.log_metric("validation NLPD", -float(val_NLPD.detach().cpu()))
        test_NLPD = NLPD(theta, x_test, y_test, sigma_noise)
        mlflow.log_metric("test NLPD", -float(test_NLPD.detach().cpu()))
        test_ib_NLPD = NLPD(theta, x_test_ib, y_test_ib, sigma_noise)
        mlflow.log_metric("test in-between NLPD", -float(test_ib_NLPD.detach().cpu()))

        
        
        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Training set')
        plt_linewidth=get_linewidth(6*sigma_noise,ax)

        for theta in ensemble:
            y_pred = mlp(x_lin, theta)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05,
                     linewidth=plt_linewidth, color='black', zorder=80)
            # res = 5
            # for r in range(res):
            #     mass = 1.0 - (r + 1) / res
            #     plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * sigma_noise * ((r + 1) / res),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * sigma_noise * ((r + 1) / res),
            #                      alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_train.cpu(), y_train.cpu(), c='red', zorder=1)
        fig.savefig(tempdir.name + '/training.png', dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/training.png')
        plt.close()

        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Validation set')
        plt_linewidth=get_linewidth(6*sigma_noise,ax)

        for theta in ensemble:
            y_pred = mlp(x_lin, theta)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05,
                     linewidth=plt_linewidth, color='black', zorder=80)
            # res = 5
            # for r in range(res):
            #     mass = 1.0 - (r + 1) / res
            #     plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * sigma_noise * ((r + 1) / res),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * sigma_noise * ((r + 1) / res),
            #                      alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_validation.cpu(), y_validation.cpu(), c='red', zorder=1)
        fig.savefig(tempdir.name + '/validation.png', dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/validation.png')
        plt.close()

        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt_linewidth=get_linewidth(2*sigma_noise,ax)

        plt.title('Test set')
        for theta in ensemble:
            y_pred = mlp(x_lin, theta)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=0.05,
                     linewidth=plt_linewidth, color='black', zorder=80)
            # res = 5
            # for r in range(res):
            #     mass = 1.0 - (r + 1) / res
            #     plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * sigma_noise * ((r + 1) / res),
            #                      y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * sigma_noise * ((r + 1) / res),
            #                      alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_test.cpu(), y_test.cpu(), c='red', zorder=1)
        fig.savefig(tempdir.name + '/test.png', dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/test.png')
        plt.close()
