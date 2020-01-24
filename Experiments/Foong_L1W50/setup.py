import torch
from torch import nn
from Tools.NNtools import *
import tempfile
import mlflow
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torch.distributions.multivariate_normal import MultivariateNormal

nblayers = 1
layerwidth = 50
param_count = 151
experiment_name = 'Foong L1/W50'

class MLP(nn.Module):
    def __init__(self, nblayers, layerwidth):
        super(MLP, self).__init__()
        L = [1] + [layerwidth]*nblayers + [1]
        self.layers = nn.ModuleList()
        for k in range(len(L)-1):
            self.layers.append(nn.Linear(L[k], L[k+1]))

    def forward(self, x):
        for j in range(len(self.layers)-1):
            x = torch.tanh(self.layers[j](x))
        x = self.layers[-1](x)
        return x

class parallel_MLP(nn.Module):
            def __init__(self, layerwidth):
                super(parallel_MLP, self).__init__()
                self.nb_neur = layerwidth
                self.activation=nn.Tanh()
                self.requires_grad_(False)
                self.param_count=3*layerwidth+1

            def forward(self,theta,x):
                nb_theta=theta.shape[0]
                theta=theta.split([self.nb_neur,self.nb_neur,self.nb_neur,1],dim=1)
                nb_x=x.shape[0]
                input_x=x.view(nb_x,1,1)
                m1=torch.matmul(theta[0].view(nb_theta,1,self.nb_neur,1),input_x)
                m2=m1.add(theta[1].reshape(nb_theta,1,self.nb_neur,1))
                m3=self.activation(m2)
                m4=torch.matmul(theta[2].view(nb_theta,1,1,self.nb_neur),m3)
                m5=m4.add(theta[3].reshape(nb_theta,1,1,1))
                return m5.squeeze(-1)

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
        loglikelihood=torch.sum(L,1)
        N=torch.as_tensor(float(theta.shape[0]),device=device)
        likelihood_mean=loglikelihood.logsumexp(0)-torch.log(N)
        return likelihood_mean
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


def log_model_evaluation(ensemble, device):
    with torch.no_grad():
        tempdir = tempfile.TemporaryDirectory(dir='/dev/shm')

        logposterior = get_logposterior_fn(device)
        model = get_model(device)
        x_train, y_train = get_training_data(device)
        x_validation, y_validation = get_validation_data(device)
        x_test, y_test = get_test_data(device)
        logtarget = lambda theta: logposterior(theta, model, x_train, y_train, 0.1)

        logposteriorpredictive = get_logposteriorpredictive_fn(device)
        train_post = logposteriorpredictive(ensemble, model, x_train, y_train, 0.1) / len(y_train)
        mlflow.log_metric("training log posterior predictive", -float(train_post.detach().cpu()))
        val_post = logposteriorpredictive(ensemble, model, x_validation, y_validation, 0.1) / len(y_validation)
        mlflow.log_metric("validation log posterior predictive", -float(val_post.detach().cpu()))
        test_post = logposteriorpredictive(ensemble, model, x_test, y_test, 0.1) / len(y_test)
        mlflow.log_metric("test log posterior predictive", -float(test_post.detach().cpu()))

        x_lin = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        plt.xlim(-2, 2)
        plt.ylim(-4, 4)
        plt.grid(True, which='major', linewidth=0.5)
        plt.title('Training set')
        for theta in ensemble:
            set_all_parameters(model, theta)
            y_pred = model(x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0,
                     linewidth=1.0, color='black', zorder=80)
            res = 5
            for r in range(res):
                mass = 1.0 - (r + 1) / res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * 0.1 * ((r + 1) / res),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * 0.1 * ((r + 1) / res),
                                 alpha=0.2 * mass, color='lightblue', zorder=50)
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
        for theta in ensemble:
            set_all_parameters(model, theta)
            y_pred = model(x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0,
                     linewidth=1.0, color='black', zorder=80)
            res = 5
            for r in range(res):
                mass = 1.0 - (r + 1) / res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * 0.1 * ((r + 1) / res),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * 0.1 * ((r + 1) / res),
                                 alpha=0.2 * mass, color='lightblue', zorder=50)
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
        for theta in ensemble:
            set_all_parameters(model, theta)
            y_pred = model(x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0,
                     linewidth=1.0, color='black', zorder=80)
            res = 5
            for r in range(res):
                mass = 1.0 - (r + 1) / res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * 0.1 * ((r + 1) / res),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * 0.1 * ((r + 1) / res),
                                 alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_test.cpu(), y_test.cpu(), c='red', zorder=100)
        fig.savefig(tempdir.name + '/test.png', dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/test.png')
        plt.close()

def log_model_evaluation_parallel(ensemble, device):
    with torch.no_grad():
        tempdir = tempfile.TemporaryDirectory(dir='/dev/shm')

        model = get_parallel_model(device)
        x_train, y_train = get_training_data(device)
        x_validation, y_validation = get_validation_data(device)
        x_test, y_test = get_test_data(device)

        ensemble_t=torch.cat(ensemble, dim=0)
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
        for theta in ensemble:
            y_pred = model(theta,x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0,
                     linewidth=1.0, color='black', zorder=80)
            res = 5
            for r in range(res):
                mass = 1.0 - (r + 1) / res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * 0.1 * ((r + 1) / res),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * 0.1 * ((r + 1) / res),
                                 alpha=0.2 * mass, color='lightblue', zorder=50)
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
        for theta in ensemble:
            y_pred = model(theta,x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0,
                     linewidth=1.0, color='black', zorder=80)
            res = 5
            for r in range(res):
                mass = 1.0 - (r + 1) / res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * 0.1 * ((r + 1) / res),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * 0.1 * ((r + 1) / res),
                                 alpha=0.2 * mass, color='lightblue', zorder=50)
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
        for theta in ensemble:
            y_pred = model(theta,x_lin)
            plt.plot(x_lin.detach().cpu().numpy(), y_pred.squeeze(0).detach().cpu().numpy(), alpha=1.0,
                     linewidth=1.0, color='black', zorder=80)
            res = 5
            for r in range(res):
                mass = 1.0 - (r + 1) / res
                plt.fill_between(x_lin.detach().cpu().numpy().squeeze(),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() - 3 * 0.1 * ((r + 1) / res),
                                 y_pred.squeeze(0).detach().cpu().numpy().squeeze() + 3 * 0.1 * ((r + 1) / res),
                                 alpha=0.2 * mass, color='lightblue', zorder=50)
        plt.scatter(x_test.cpu(), y_test.cpu(), c='red', zorder=100)
        fig.savefig(tempdir.name + '/test.png', dpi=4 * fig.dpi)
        mlflow.log_artifact(tempdir.name + '/test.png')
        plt.close()
