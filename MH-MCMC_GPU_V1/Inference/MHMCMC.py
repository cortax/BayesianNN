import numpy as np
from scipy.stats import norm, bayes_mvs, multivariate_normal
import emcee
import torch
from torch import distributions as dist
import matplotlib.pyplot as plt
from torch import nn
import time



class netModel(nn.Module):
    
    def __init__(self, nNeurons, device=None):
        super(netModel, self).__init__()
        
        self.device = device
            
        self.linear1 = nn.Linear(1, nNeurons)
        self.linear2 = nn.Linear(nNeurons, 1)
        
        self.to(self.device)
        
    def forward(self, x):

        out = torch.tanh(self.linear1(x.unsqueeze(-1)).squeeze(1))
        out = self.linear2(out)
        
        return out

class MCNN():
    
    def __init__(self, model, nNeurons, nLayers):
        
        self.model = model
        self.device = model.device 
        self.nNeurons = nNeurons
        self.nLayers = nLayers
        
        self.nParameters = self.nNeurons * self.nLayers + self.nNeurons * (self.nLayers - 1) + 1

    
    def updateParameters(self, position):
        
        linear1_w = position[:self.nNeurons]
        linear1_b = position[self.nNeurons:2*self.nNeurons]
        linear2_w = position[2*self.nNeurons:3*self.nNeurons]
        linear2_b = position[3*self.nNeurons:]

        linear1_w = torch.FloatTensor(linear1_w).float().view(self.nNeurons, 1).to(self.device)
        linear1_b = torch.FloatTensor(linear1_b).float().to(self.device)
        linear2_w = torch.FloatTensor(linear2_w).float().view(1, self.nNeurons).to(self.device)
        linear2_b = torch.FloatTensor(linear2_b).float().to(self.device)

        parameters = [linear1_w, linear1_b, linear2_w, linear2_b]    

        i = 0

        for param in self.model.parameters():

            param.data = parameters[i]

            i = i + 1

    def computeLogPosterior(self, w, x, y):
        
        self.updateParameters(w)

        y_p = self.model(x).detach()
        y_p = y_p.squeeze(-1)

        Normal1 = dist.Normal(y_p.detach(), 0.1)
        Normal2 = dist.MultivariateNormal(torch.zeros(self.nParameters),  torch.eye(self.nParameters))
        log_l = torch.sum(Normal1.log_prob(y))
        log_p =  Normal2.log_prob(torch.FloatTensor(w))
        B = log_l.item() + log_p.item()

        return B

    def runMHMCMC(self, data, nBurnin, nSamples, thin):
        
        x_data = data[0].to(self.device)
        y_data = data[1].to(self.device)
        y_data = y_data.unsqueeze(-1)

        p0 = np.random.rand(self.nParameters)+2
        
        cov = np.identity(self.nParameters)*0.00001

        sampler = emcee.MHSampler(cov, self.nParameters, self.computeLogPosterior, args=[x_data, y_data])
        pos, prob, state = sampler.run_mcmc(p0, nBurnin)
        sampler.reset()

        kwargs = {'thin':thin}
        sampler.run_mcmc(pos, nSamples, **kwargs)

        return sampler
    
def plot_MCMC_loss(sampler):
    N = sampler.flatchain.shape[0]
    logProb = []
    for sample in range(50):
        logProb.append(sampler.get_lnprob(sampler.flatchain[sample]))
    plt.plot(logProb)

def updateParameters_plot(model, position, nNeurons, device):

    linear1_w = position[:nNeurons]
    linear1_b = position[nNeurons:2*nNeurons]
    linear2_w = position[2*nNeurons:3*nNeurons]
    linear2_b = position[3*nNeurons:]

    linear1_w = torch.FloatTensor(linear1_w).float().view(nNeurons, 1).to(device)
    linear1_b = torch.FloatTensor(linear1_b).float().to(device)
    linear2_w = torch.FloatTensor(linear2_w).float().view(1, nNeurons).to(device)
    linear2_b = torch.FloatTensor(linear2_b).float().to(device)

    parameters = [linear1_w, linear1_b, linear2_w, linear2_b]    

    i = 0

    for param in model.parameters():

        param.data = parameters[i]

        i = i + 1
        
def plot_MCMC(model, nNeurons, sampler, data, device, savePath=None, networkName=None, saveName=None):
    
    x_test = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 9)
    plt.axis([-2, 2, -2, 3])
    plt.scatter(data[0].cpu(), data[1].cpu())
    
    y = torch.cos(4.0*(x_test+0.2))
    
    plt.plot(x_test.cpu().numpy(), y.cpu().numpy())
    
    N = sampler.flatchain.shape[0]
    
    for sample in range(N): 
        updateParameters_plot(model, sampler.flatchain[sample], nNeurons, device)
        y_test = model.forward(x_test.to(device))

        plt.plot(x_test.cpu().numpy(), y_test.detach().cpu().numpy(), alpha=0.05, linewidth=1.0,color='lightblue')
        
    plt.savefig(savePath + networkName +'_'+ saveName)
    