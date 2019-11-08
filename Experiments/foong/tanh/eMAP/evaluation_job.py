import sys
import os
from os.path import dirname
cwd = os.path.dirname(os.path.realpath(__file__))
rootdir = dirname(dirname(dirname(dirname(cwd))))
sys.path.append( rootdir )

cwd_MAP =  rootdir + '/Experiments/foong/tanh/MAP'

from Inference import BBVI 
from Inference.VariationalBoosting import MixtureVariationalNetwork
from Inference.VariationalBoosting import VariationalBoostingOptimizer

import torch
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')

def plot_ensemble(mix, MAPs_type, mix_name, K):
    device = mix.device
    
    for k in range(K):
	    filename = cwd_MAP + '/models/'+ MAPs_type + '('+ str(k) + ')'
	    filehandler = open(filename, 'rb')
	    netparam = pickle.load(filehandler)
	    Net = BBVI.VariationalNetwork(input_size=netparam['input_size'],
		                      output_size=netparam['output_size'],
		                      layer_width=netparam['layer_width'],
		                      nb_layers=netparam['nb_layers'])
	    Net.set_network(netparam)
	    Net.set_device(device)
	    mix.add_component(Net, torch.tensor(1.0/(k+1.0)))
    
    data = torch.load(rootdir + '/Data/foong_data.pt')
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = y_data.unsqueeze(-1)

    data = torch.load(rootdir +'/Data/foong_data_validation.pt')
    x_data_validation = data[0].to(device)
    y_data_validation = data[1].to(device)
    y_data_validation = y_data_validation.unsqueeze(-1)
    
    data = torch.load(rootdir +'/Data/foong_data_test.pt')
    x_data_test = data[0].to(device)
    y_data_test = data[1].to(device)
    y_data_test = y_data_test.unsqueeze(-1)

    x_linspace = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)

    mix.sample_parameters(1000)
    y_train = mix.forward(x_data).to(device) 
        
    y_validation = mix.forward(x_data_validation).to(device)
    y_test = mix.forward(x_data_test).to(device)
    
    ELL_train = mix._log_norm(y_train, y_data, torch.tensor(0.1).to(device))
    ELL_train = ELL_train.sum(dim=[1,2]).mean().detach().cpu().numpy()

        
    ELL_val = mix._log_norm(y_validation, y_data_validation, torch.tensor(0.1).to(device))
    ELL_val = ELL_val.sum(dim=[1,2]).mean().detach().cpu().numpy()
    
    ELL_test = mix._log_norm(y_test, y_data_test, torch.tensor(0.1).to(device))
    ELL_test = ELL_test.sum(dim=[1,2]).mean().detach().cpu().numpy()

    ELBO_train = mix.compute_elbo(x_data, y_data, n_samples_ELBO=1000, sigma_noise=0.1).detach().cpu().numpy()
    ELBO_val = mix.compute_elbo(x_data_validation, y_data_validation, n_samples_ELBO=1000, sigma_noise=0.1).detach().cpu().numpy()
    ELBO_test = mix.compute_elbo(x_data_test, y_data_test, n_samples_ELBO=1000, sigma_noise=0.1).detach().cpu().numpy()
    
    INFOS = [ELL_train, ELL_val, ELL_test, ELBO_train, ELBO_val, ELBO_test]
    
    y_real = torch.cos(4.0 * (x_linspace + 0.2))
    
    print('plotting...')

    fig = plt.figure() 

    plt.axis([-2, 2, -2, 3.5])
    plt.scatter(x_data.clone().detach().cpu().numpy(), y_data.clone().detach().cpu().numpy(), s=3)
    plt.plot(x_linspace.clone().detach().cpu().numpy(), y_real.clone().detach().cpu().numpy())

    for _ in range(1000):

        mix.sample_parameters()

        y_test_PD = mix.forward(x_linspace)
        
        plt.plot(x_linspace.detach().cpu().numpy(), y_test_PD.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='lightblue')
        
    plt.text(-1.7, 3, 'Expected log-likelihood (train): '+str(ELL_train), fontsize=10)
    plt.text(-1.7, 2.75, 'Expected log-likelihood (val): '+str(ELL_val), fontsize=10)
    plt.text(-1.7, 2.5, 'Expected log-likelihood (test): '+str(ELL_test), fontsize=10)
    plt.text(-1.7, 2.25, 'ELBO (train): '+ str(ELBO_train), fontsize=10)
    plt.text(-1.7, 2, 'ELBO (val): '+ str(ELBO_val), fontsize=10)
    plt.text(-1.7, 1.75, 'ELBO (test): '+ str(ELBO_test), fontsize=10)
   
    plt.savefig(cwd + '/plots/' + mix_name)

    plt.close(fig)

    return INFOS
    
def update_log(mix_name, INFOS):
    
    log = open(cwd + '/logs/' + mix_name + '.txt', 'a+')

    log.write('Expected log-likelihood on the train set:' + str(INFOS[0]) + '\n')
    log.write('Expected log-likelihood on the validation set:' + str(INFOS[1]) + '\n')
    log.write('Expected log-likelihood on the test set:' + str(INFOS[2]) + '\n')

    log.write('ELBO on the train set:' + str(INFOS[3]) + '\n')
    log.write('ELBO on the train set:' + str(INFOS[4]) + '\n')
    log.write('ELBO on the train set:' + str(INFOS[5]) + '\n')
    
    log.close()
    
if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    activation = torch.tanh

    idx = np.random.permutation(150)
    j = idx[0]

    K = 150
  
    L = 4
    W = 50
    MAPs_type = str(L) + 'Layers_' +str(W)+'Neurons_'

    filename = cwd_MAP + '/models/' + MAPs_type + '(' +  str(j) + ')'
    filehandler = open(filename, 'rb')
    netparam = pickle.load(filehandler)
    mix_name = str(K) +'ensemble_' + MAPs_type + '(' + str(j)+ ')'
    mix = MixtureVariationalNetwork(netparam['input_size'], netparam['output_size'], netparam['layer_width'], netparam['nb_layers'], device=device)

    INFOS = plot_ensemble(mix, MAPs_type, mix_name, K)
    update_log(mix_name, INFOS)
