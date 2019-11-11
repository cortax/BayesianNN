import numpy as np
import math
import torch
from torch import nn
from torch import functional as F

from livelossplot import PlotLosses

import matplotlib.pyplot as plt
plt.style.use('ggplot')

class ProbabilisticLinear(nn.Module):

    def __init__(self, in_features, out_features, device=None, bias=True):
        super(ProbabilisticLinear, self).__init__()
        
        self.device = device
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.q_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.q_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.q_bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.q_bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.weight_sample = torch.empty([0, out_features, in_features], device=device)
        self.bias_sample = torch.empty([0, out_features, in_features], device=device)
        
        self.reset_parameters()
        
        mu = torch.tensor(0.0)
        sigma = torch.tensor(1.0)
        rho = torch.log(torch.exp(sigma) - 1)
        
        torch.nn.init.normal_(self.q_weight_mu, mean=mu, std=sigma/out_features)
        torch.nn.init.normal_(self.q_weight_rho, mean=mu, std=sigma/out_features)
        torch.nn.init.normal_(self.q_bias_mu, mean=mu, std=sigma/out_features)
        torch.nn.init.normal_(self.q_bias_rho, mean=mu, std=sigma/out_features)
        
        self.prior_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.prior_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.prior_bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.prior_bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.prior_weight_mu.requires_grad = False
        self.prior_weight_rho.requires_grad = False
        self.prior_bias_mu.requires_grad = False
        self.prior_bias_rho.requires_grad = False
        
        nn.init.constant_(self.prior_weight_mu, mu)
        nn.init.constant_(self.prior_weight_rho, rho)
        nn.init.constant_(self.prior_bias_mu, mu)
        nn.init.constant_(self.prior_bias_rho, rho)
        
        self.to(self.device)

    def _rho_to_sigma(self, rho):
        sigma = torch.log(1 + torch.exp(rho))
        return sigma

    def _sigma_to_rho(self, sigma):
        rho = torch.log(torch.exp(sigma) - 1)
        return rho

    def sample_parameters(self, M=1):
        size = [M]+list(self.q_weight_mu.size())
        weight_epsilon = torch.randn(size=size).to(self.device)
        size = [M]+list(self.q_bias_mu.size())
        bias_epsilon = torch.randn(size=size).to(self.device)
    
        sigma_weight = self._rho_to_sigma(self.q_weight_rho)
        sigma_bias = self._rho_to_sigma(self.q_bias_rho)
        
        if M > 0:
            self.weight_sample = weight_epsilon.mul(sigma_weight.unsqueeze(0)).add(self.q_weight_mu.unsqueeze(0).repeat(M,1,1))
            self.bias_sample = bias_epsilon.mul(sigma_bias.unsqueeze(0)).add(self.q_bias_mu.unsqueeze(0).repeat(M,1))
        else:
            self.weight_sample = weight_epsilon.mul(sigma_weight.unsqueeze(0))
            self.bias_sample = bias_epsilon.mul(sigma_bias.unsqueeze(0))
        
        return (self.weight_sample, self.bias_sample)

    def requires_grad_rhos(self, v = False):
        self.q_weight_rho.requires_grad = v
        self.q_bias_rho.requires_grad = v
        
    def requires_grad_mus(self, v = False):
        self.q_weight_mu.requires_grad = v
        self.q_bias_mu.requires_grad = v
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.q_weight_mu, mean=0.0, std=5.0)
        torch.nn.init.constant_(self.q_weight_rho, -1.0)
        torch.nn.init.normal_(self.q_bias_mu, mean=0.0, std=5.0)
        torch.nn.init.constant_(self.q_bias_rho, -1.0)
       
    def forward(self, x):
        M = self.weight_sample.size(0)
        if x.dim() < 3:
            if M > 0:
                x = x.unsqueeze(0).repeat(M,1,1)
            else:
                s = [0] + list(x.shape)
                x = torch.empty(s, device=self.device)
        if M > 0:
            return x.matmul(self.weight_sample.permute(0,2,1)).add(self.bias_sample.unsqueeze(-1).permute(0,2,1))
        else:
            return x.matmul(self.weight_sample.permute(0,2,1))

    
class VariationalNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_width, nb_layers, activation=torch.tanh, device=None):
        super(VariationalNetwork, self).__init__()
        
        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size
        self.layer_width = layer_width
        self.nb_layers = nb_layers
        
        self.device = device
        self.registered_layers = []

        # Dynamically creating layers and registering them as parameters to optimize 
        layer_dims = [input_size] + [layer_width] * (nb_layers-1) + [output_size]
        for k in range(nb_layers):
            a = layer_dims[k]
            b = layer_dims[k+1]
            layer = ProbabilisticLinear(a, b, device)
            exec('self.linear' + str(k+1) + ' = layer')
            self.registered_layers.append(layer)
        
        self.nb_parameters = self.count_parameters()
        self.to(device)
        
    def _log_norm(self, x, mu, std):
        return -0.5 * torch.log(2*np.pi*std**2) -(0.5 * (1/(std**2))* (x-mu)**2)
    
    def _rho_to_sigma(self, rho):
        sigma = torch.log(1 + torch.exp(rho))
        return sigma

    def _sigma_to_rho(self, sigma):
        rho = torch.log(torch.exp(sigma) - 1)
        return rho
              
    def forward(self, x):
        out = x
        for k in range(len(self.registered_layers)-1):
            out = self.activation(self.registered_layers[k](out))
        out = self.registered_layers[-1](out)
        return out
    
    def requires_grad_rhos(self, v = False):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].requires_grad_rhos(v)
        
    def requires_grad_mus(self, v = False):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].requires_grad_mus(v)
            
    def make_deterministic_rhos(self, v=-10.0):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].q_weight_rho = nn.Parameter(self.registered_layers[k].q_weight_rho.new_full(self.registered_layers[k].q_weight_rho.size(), v))
            self.registered_layers[k].q_bias_rho = nn.Parameter(self.registered_layers[k].q_bias_rho.new_full(self.registered_layers[k].q_bias_rho.size(), v))
    
    def sample_parameters(self, M=1):
        layered_w_samples = []
        layered_bias_samples = []
        L = [self.registered_layers[k].sample_parameters(M) for k in range(len(self.registered_layers))]
        for k in range(len(self.registered_layers)):
            if L[k] is not None:
                layered_w_samples.append(L[k][0])
                layered_bias_samples.append(L[k][1])
        return (layered_w_samples, layered_bias_samples)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_parameters(self, w_samples, b_samples):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].set_parameters(w_samples[k], b_samples[k])
            
    def get_parameters(self):
        for k in range(len(self.registered_layers)):
            self.registered_layers[k].set_parameters(w_samples[k], b_samples[k]) 
        
        return w_samples, b_samples
    
    def KL_log_pdf(self, w_samples, b_samples):
        self.set_parameters(w_samples, b_samples)
        return (self.q_log_pdf(), self.prior_log_pdf())

    def q_log_pdf(self, layered_w_samples, layered_bias_samples):
        mu = torch.cat([torch.cat((torch.flatten(self.registered_layers[k].q_weight_mu), self.registered_layers[k].q_bias_mu)) for k in range(len(self.registered_layers))])
        rho = torch.cat([torch.cat((torch.flatten(self.registered_layers[k].q_weight_rho), self.registered_layers[k].q_bias_rho)) for k in range(len(self.registered_layers))])
        sigma = self._rho_to_sigma(rho)
        nw = torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(sigma))
        w = torch.cat([torch.cat((torch.flatten(layered_w_samples[k], start_dim=1), layered_bias_samples[k]),dim=1) for k in range(len(self.registered_layers))], dim=1)
        return nw.log_prob(w)
    
    def prior_log_pdf(self, layered_w_samples, layered_bias_samples):
        mu = torch.cat([torch.cat((torch.flatten(self.registered_layers[k].prior_weight_mu), self.registered_layers[k].prior_bias_mu)) for k in range(len(self.registered_layers))])
        rho = torch.cat([torch.cat((torch.flatten(self.registered_layers[k].prior_weight_rho), self.registered_layers[k].prior_bias_rho)) for k in range(len(self.registered_layers))])
        sigma = self._rho_to_sigma(rho)
        nw = torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(sigma))
        w = torch.cat([torch.cat((torch.flatten(layered_w_samples[k], start_dim=1), layered_bias_samples[k]),dim=1) for k in range(len(self.registered_layers))], dim=1)
        return nw.log_prob(w)
        
    def compute_elbo(self, x_data, y_data, n_samples_ELBO, sigma_noise, device):
        (layered_w_samples, layered_bias_samples) = self.sample_parameters(n_samples_ELBO)

        LQ = self.q_log_pdf(layered_w_samples, layered_bias_samples)
        LP = self.prior_log_pdf(layered_w_samples, layered_bias_samples)
        MCKL = (LQ - LP).mean()
        
        y_pred = self.forward(x_data)
        mu = torch.flatten(y_pred, end_dim=1)
        sigma = torch.eye(y_data.shape[1],device=device)*torch.tensor(sigma_noise, device=device)
        nd = torch.distributions.MultivariateNormal(mu, scale_tril=sigma)
        LL = nd.log_prob(torch.flatten(y_data.repeat(y_pred.shape[0], 1, 1), end_dim=1))
        eLL = LL.reshape(y_pred.shape[0:2]).sum(1).mean()
        
        loss = MCKL - eLL
        
        return loss
    
    def get_network(self):
        network = {}
        
        network['activation'] = self.activation
        network['input_size'] = self.input_size
        network['output_size'] = self.output_size
        network['layer_width'] = self.layer_width
        network['nb_layers'] = self.nb_layers
        
        registered_layers = []

        for L in self.registered_layers:
            d = {}
            d['q_weight_mu'] = L.q_weight_mu.to('cpu')
            d['q_weight_rho'] = L.q_weight_rho.to('cpu')
            d['q_bias_mu'] = L.q_bias_mu.to('cpu')
            d['q_bias_rho'] = L.q_bias_rho.to('cpu')
            registered_layers.append(d)
        network['registered_layers'] = registered_layers
        
        return network
    
    def set_network(self, network):
        self.activation = network['activation']
        self.input_size = network['input_size'] 
        self.output_size = network['output_size']
        self.layer_width = network['layer_width']
        self.nb_layers = network['nb_layers']
        
        registered_layers = []

        for i, d in enumerate(network['registered_layers']):
            self.registered_layers[i].q_weight_mu = torch.nn.Parameter(d['q_weight_mu'])
            self.registered_layers[i].q_weight_rho = torch.nn.Parameter(d['q_weight_rho'])
            self.registered_layers[i].q_bias_mu = torch.nn.Parameter(d['q_bias_mu'])
            self.registered_layers[i].q_bias_rho = torch.nn.Parameter(d['q_bias_rho'])
            
    def set_device(self, device):
        self.device = device
        for i in range(len(self.registered_layers)):
            self.registered_layers[i].device = device
            self.registered_layers[i].q_weight_mu = torch.nn.Parameter(self.registered_layers[i].q_weight_mu.to(device))
            self.registered_layers[i].q_weight_rho = torch.nn.Parameter(self.registered_layers[i].q_weight_rho.to(device))
            self.registered_layers[i].q_bias_mu = torch.nn.Parameter(self.registered_layers[i].q_bias_mu.to(device))
            self.registered_layers[i].q_bias_rho = torch.nn.Parameter(self.registered_layers[i].q_bias_rho.to(device))

            self.registered_layers[i].prior_weight_mu = torch.nn.Parameter(self.registered_layers[i].prior_weight_mu.to(device), requires_grad=False)
            self.registered_layers[i].prior_weight_rho = torch.nn.Parameter(self.registered_layers[i].prior_weight_rho.to(device), requires_grad=False)
            self.registered_layers[i].prior_bias_mu = torch.nn.Parameter(self.registered_layers[i].prior_bias_mu.to(device), requires_grad=False)
            self.registered_layers[i].prior_bias_rho = torch.nn.Parameter(self.registered_layers[i].prior_bias_rho.to(device), requires_grad=False)


class VariationalOptimizer():
    def __init__(self, model, sigma_noise, optimizer, optimizer_params, scheduler=None, scheduler_params=None, min_lr=None):
        self.model = model
        self.sigma_noise = sigma_noise
        self.device = model.device
        self.min_lr = min_lr
        
        self.last_epoch = None
        
        self.optimizer = optimizer(model.parameters(), **optimizer_params)
        if scheduler is None:           
            self.scheduler = None
        else:
            self.scheduler = scheduler(self.optimizer, **scheduler_params)
            
    def run(self, data, n_epoch=100, n_iter=1, n_ELBO_samples=1, seed=None, plot=False, log=False, verbose=0):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x_data = data[0].to(self.device)
        y_data = data[1].to(self.device)

        self.optimizer.zero_grad()

        if plot:
            liveloss = PlotLosses()

        if log:
            self.log = []

        for j in range(n_epoch):
            logs = {}
            losses = [None] * n_iter

            for k in range(n_iter):
                self.optimizer.zero_grad()
                loss = self.model.compute_elbo(x_data, y_data, n_ELBO_samples, self.sigma_noise, self.device)
                losses[k] = loss
                loss.backward()
                self.optimizer.step()

            expected_loss = torch.stack(losses).mean().detach().clone().cpu().numpy()
            learning_rate = self.optimizer.param_groups[0]['lr']

            if plot:
                logs['expected_loss'] = expected_loss
                logs['learning rate'] = learning_rate
                liveloss.update(logs)
                liveloss.draw()

            if verbose >= 1:
                print("Epoch: {0:6d} / training loss: {1:16.3f} / learning rate: {2:1.7f}".format(j,expected_loss,learning_rate))

            if log:
                self.log.append("Epoch: {0:6d} / training loss: {1:16.3f} / learning rate: {2:1.7f}".format(j,expected_loss,learning_rate))

            if self.scheduler is not None:
                self.scheduler.step(expected_loss)
                if self.min_lr is not None and learning_rate < self.min_lr:
                    self.last_epoch = j
                    return self.model
        self.last_epoch = n_epoch    
        return self.model
    
    
class MCMCsimulator():
    def __init__(self, model):
        self.model = model
        self.model.sample_parameters(2);
    
       
    def MetropolisHastings(self, x_data, y_data, nb_iter, sigma_proposal = 0.001, plot=False):
        acceptance = []
        samples = []
        proba = []
        
        if plot:
            liveloss = PlotLosses()
        
        with torch.no_grad():
            for t in range(nb_iter):
                logs = {}
                #proposal 
                for layer in self.model.registered_layers:
                    layer.weight_sample[1,:,:].normal_(mean=0, std=sigma_proposal)
                    layer.weight_sample[1,:,:].add_(layer.weight_sample[0,:,:])

                    layer.bias_sample[1,:].normal_(mean=0, std=sigma_proposal)
                    layer.bias_sample[1,:].add_(layer.bias_sample[0,:])

                #proba
                mu = torch.cat([torch.cat((torch.flatten(self.model.registered_layers[k].prior_weight_mu), self.model.registered_layers[k].prior_bias_mu)) for k in range(len(self.model.registered_layers))])
                rho = torch.cat([torch.cat((torch.flatten(self.model.registered_layers[k].prior_weight_rho), self.model.registered_layers[k].prior_bias_rho)) for k in range(len(self.model.registered_layers))])
                sigma = self.model._rho_to_sigma(rho)
                nw = torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(sigma))
                w0 = torch.cat([torch.cat((torch.flatten(self.model.registered_layers[k].weight_sample[0]), self.model.registered_layers[k].bias_sample[0])) for k in range(len(self.model.registered_layers))])
                w1 = torch.cat([torch.cat((torch.flatten(self.model.registered_layers[k].weight_sample[1]), self.model.registered_layers[k].bias_sample[1])) for k in range(len(self.model.registered_layers))])
                LP0 = nw.log_prob(w0)
                LP1 = nw.log_prob(w1)

                y_pred = self.model.forward(x_data)
                LL0 = self.model._log_norm(y_pred[0], y_data, torch.tensor(0.1)).sum()
                LL1 = self.model._log_norm(y_pred[1], y_data, torch.tensor(0.1)).sum()


                A = (LL1 + LP1) - (LL0 + LP0)
                if torch.distributions.uniform.Uniform(0.0,1.0).sample().log() < A.cpu():
                    acceptance.append(1)
                    samples.append(w1.detach().clone().cpu())
                    proba.append(LL1 + LP1) 
                    logs['log proba'] = (LL1 + LP1).detach().clone().cpu().numpy()
                    
                    for layer in self.model.registered_layers:
                        layer.weight_sample[0,:,:] = layer.weight_sample[1,:,:].detach()
                        layer.bias_sample[0,:] = layer.bias_sample[1,:].detach()
                else:
                    acceptance.append(0)
                    samples.append(w0.detach().clone().cpu())
                    proba.append(LL0 + LP0)
                    logs['log proba'] = (LL0 + LP0).detach().clone().cpu().numpy()
                
                if plot:
                    liveloss.update(logs)
                    liveloss.draw()

        return samples, acceptance, proba
        
    
def plot_BBVI(model, data, data_val, device, savePath=None, xpName=None, networkName=None, saveName=None):
    
    x_test = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 9)
    plt.axis([-2, 2, -2, 3])
    plt.scatter(data[0].cpu(), data[1].cpu())
    
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = y_data.unsqueeze(-1)

    x_data_validation = data_val[0].to(device)
    y_data_validation = data_val[1].to(device)
    y_data_validation = y_data_validation.unsqueeze(-1)
    
    y_pred_validation = model.forward(x_data_validation)
    
    LL = model._log_norm(y_pred_validation, y_data_validation, torch.tensor(0.1).to(device))
    LL = LL.sum(dim=[1,2]).mean().detach().cpu().numpy()
    
    L_ELBO = model.compute_elbo(x_data, y_data, n_samples_ELBO=2000, sigma_noise=0.1, device=device).detach().cpu().numpy()
    
    
    y = torch.cos(4.0*(x_test+0.2))
    plt.plot(x_test.cpu().detach().clone().numpy(), y.cpu().detach().clone().numpy())
    for _ in range(1500):
        
        model.sample_parameters()

        y_test = model.forward(x_test)
        plt.plot(x_test.detach().cpu().numpy(), y_test.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='lightblue')
    plt.text(-1.7, 2, 'Expected Loglikelihood:'+str(LL), fontsize=12)
    plt.text(-1.7, 2.5, 'Last ELBO:'+str(L_ELBO), fontsize=12)
  

    plt.savefig(savePath + saveName +'_'+ networkName)
    
