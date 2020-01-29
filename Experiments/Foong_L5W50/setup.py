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


nb_layers = 5
layerwidth = 100
param_count = 40701
experiment_name = 'Foong L4/W200'


class parallel_MLP(nn.Module):
            def __init__(self, layerwidth,nb_layers=1):
                super(parallel_MLP, self).__init__()
                self.layerwidth = layerwidth
                self.activation=nn.Tanh()# use identity for debugging
                self.requires_grad_(False)
                self.nb_layers=nb_layers
                self.param_count=2*layerwidth+(nb_layers-1)*(layerwidth**2+layerwidth)+layerwidth+1
                self.splitting=[layerwidth]*2+[layerwidth**2,layerwidth]*(nb_layers-1)+[layerwidth,1]
                
            def forward(self,theta,x):
                nb_theta=theta.shape[0]
                nb_x=x.shape[0]
                theta=theta.split(self.splitting,dim=1)
                input_x=x.view(nb_x,1,1)
                m=torch.matmul(theta[0].view(nb_theta,1,self.layerwidth,1),input_x)
                m=m.add(theta[1].reshape(nb_theta,1,self.layerwidth,1))
                m=self.activation(m)
                for i in range(self.nb_layers-1):
                    m=torch.matmul(theta[2*i+2].view(-1,1,self.layerwidth,self.layerwidth),m)
                    m=m.add(theta[2*i+3].reshape(-1,1,self.layerwidth,1))
                    m=self.activation(m)
                m=torch.matmul(theta[2*(self.nb_layers-1)+2].view(nb_theta,1,1,self.layerwidth),m)
                m=m.add(theta[2*(self.nb_layers-1)+3].reshape(nb_theta,1,1,1))
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

def get_model(device):
    model = MLP(nblayers, layerwidth).to(device)
    flip_parameters_to_tensors(model)
    return model

def get_parallel_model(device):
    model = parallel_MLP(layerwidth).to(device)
    return model

def _log_norm(x, mu, std):
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
    def loglikelihood(theta, model, x, y, sigma_noise):
        set_all_parameters(model, theta)
        y_pred = model(x)
        L = _log_norm(y_pred, y, torch.tensor([sigma_noise],device=device))
        return torch.sum(L).unsqueeze(-1)
    return loglikelihood

# added parallelized versions
def get_loglikelihood_parallel_fn(device):
    def loglikelihood_parallel(theta, model, x, y, sigma_noise):
        y_pred = model(theta,x)
        L = _log_norm(y_pred, y, torch.tensor([sigma_noise],device=device))
        return torch.sum(L,1)
    return loglikelihood_parallel

# added parallelized versions
def get_logposterior_parallel_fn(device):
    logprior = get_logprior_fn(device)
    loglikelihood_parallel = get_loglikelihood_parallel_fn(device)
    def logposterior_parallel(theta, model, x, y, sigma_noise):
        return logprior(theta).add(loglikelihood_parallel(theta, model, x, y, sigma_noise))
    return logposterior_parallel


def get_logposterior_fn(device):
    logprior = get_logprior_fn(device)
    loglikelihood = get_loglikelihood_fn(device)
    def logposterior(theta, model, x, y, sigma_noise):
        return logprior(theta) + loglikelihood(theta, model, x, y, sigma_noise)
    return logposterior

def get_logposteriorpredictive_parallel_fn(device):    
    def logposteriorpredictive(theta, model, x, y, sigma_noise):
        y_pred = model(theta,x)
        L = _log_norm(y_pred, y, torch.tensor([sigma_noise],device=device))
        n_x=torch.as_tensor(float(x.shape[0]),device=device)
        n_theta=torch.as_tensor(float(theta.shape[0]),device=device)
        log_posterior_predictive=torch.logsumexp(L,0)-torch.log(n_theta)
#        log_mean_posterior_predictive=log_posterior_predictive.logsumexp(0)-torch.log(n_x)
        return log_posterior_predictive.sum()
    return logposteriorpredictive

def get_logposteriorpredictive_fn(device):
    def logposteriorpredictive(ensemble, model, x, y, sigma_noise):
        complogproba = []
        for theta in ensemble:
            set_all_parameters(model, theta)
            y_pred = model(x)
            complogproba.append(-torch.tensor(float(len(ensemble))).log() + _log_norm(y_pred, y, torch.tensor([sigma_noise],device=device)))
        return torch.logsumexp(torch.stack(complogproba), dim=0).sum().unsqueeze(-1)
    return logposteriorpredictive



def log_model_evaluation_parallel(ensemble, device):
    with torch.no_grad():
        tempdir = tempfile.TemporaryDirectory(dir='/dev/shm')

        model = get_parallel_model(device)
        x_train, y_train = get_training_data(device)
        x_validation, y_validation = get_validation_data(device)
        x_test, y_test = get_test_data(device)
        x_test_ib, y_test_ib = get_test_ib_data(device)

        
       
        logposteriorpredictive = get_logposteriorpredictive_parallel_fn(device)
        train_post = logposteriorpredictive(ensemble_t, model, x_train, y_train, 0.1)
        mlflow.log_metric("training log posterior predictive", -float(train_post.detach().cpu()))
        val_post = logposteriorpredictive(ensemble_t, model, x_validation, y_validation, 0.1)     
        mlflow.log_metric("validation log posterior predictive", -float(val_post.detach().cpu()))
        test_post = logposteriorpredictive(ensemble_t, model, x_test, y_test, 0.1)
        mlflow.log_metric("test log posterior predictive", -float(test_post.detach().cpu()))

        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Training set')
        for i in range(ensemble.shape[0]):
            y_pred = model(ensemble[i],x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0,
                     linewidth=1.0, color='black', zorder=80)
            res = 5
            for r in range(res):
                mass = 1.0 - (r + 1) / res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * 0.1 * ((r + 1) / res),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * 0.1 * ((r + 1) / res),
                                 alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_train.cpu(), y_train.cpu(), c='red', zorder=1)
        fig.savefig(tempdir.name + '/training.png',dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/training.png')
        plt.close()


        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Test set')
        for i in range(ensemble.shape[0]):
            y_pred = model(ensemble[i],x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0,
                     linewidth=1.0, color='black', zorder=80)
            res = 5
            for r in range(res):
                mass = 1.0 - (r + 1) / res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * 0.1 * ((r + 1) / res),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * 0.1 * ((r + 1) / res),
                                 alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_test.cpu(), y_test.cpu(), c='red', zorder=1)
        fig.savefig(tempdir.name + '/test.png', dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/test.png')
        plt.close()
