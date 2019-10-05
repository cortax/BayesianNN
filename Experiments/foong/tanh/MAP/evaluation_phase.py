import sys
import os
from os.path import dirname
cwd = os.path.dirname(os.path.realpath(__file__))
rootdir = dirname(dirname(dirname(dirname(cwd))))
sys.path.append( rootdir )


from Inference import BBVI 
import _pickle as pickle
import torch
import matplotlib.pyplot as plt

def plot_validation(model, model_name):
        
    y_validation = model.forward(x_data_validation)

    ELL = model._log_norm(y_validation, y_data_validation, torch.tensor(0.1).to(device))
    ELL = ELL.sum(dim=[1,2]).mean().detach().cpu().numpy()

    ELBO = model.compute_elbo(x_data, y_data, n_samples_ELBO=1000, sigma_noise=0.1, device=device).detach().cpu().numpy()

    y_real = torch.cos(4.0 * (x_linspace + 0.2))
    
    plt.axis([-2, 2, -2, 3])
    plt.scatter(x_data.cpu(), y_data.cpu(), s=3)
    plt.plot(x_linspace.cpu(), y_real.cpu())

    for _ in range(1000):

        model.sample_parameters()

        y_validation_PD = model.forward(x_linspace)
        
        plt.plot(x_linspace.detach().cpu().numpy(), y_validation_PD.squeeze(0).detach().cpu().numpy(), alpha=0.05, linewidth=1, color='lightblue')
        
    plt.text(-1.7, 2, 'Expected log-likelihood (val): '+str(ELL), fontsize=11)
    plt.text(-1.7, 2.5, 'ELBO (train): '+ str(ELBO), fontsize=11)

    plt.savefig(cwd + '/plots/' + model_name)

    return ELL, ELBO
    
    
def update_log(model, model_name, ell, elbo):
    
    log = open(cwd + '/logs/' + model_name + '.txt', 'a+')

    log.write('Expected log-likelihood on the validation set:' + str(ell) + '\n')
    log.write('ELBO on the train set:' + str(elbo) + '\n')
    
    log.close()
  
        
        
        
if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    activation = torch.tanh

    layer_size = [2]
    nb_layer = [2]
    nb_trial = 1
    
#    layer_size = [10,25,50,100]
#    nb_layer = [2,3,4]
#    nb_trial = 10
    
    data = torch.load(rootdir + '/Data/foong_data.pt')
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = y_data.unsqueeze(-1)

    data = torch.load(rootdir +'/Data/foong_data_validation.pt')
    x_data_validation = data[0].to(device)
    y_data_validation = data[1].to(device)
    y_data_validation = y_data_validation.unsqueeze(-1)

    x_linspace = torch.linspace(-2.0, 2.0).unsqueeze(1).to(device)

    for L in nb_layer:
        for W in layer_size:
            for j in range(nb_trial):
                Net_name = str(L) + 'Layers_' + str(W) + 'Neurons_(' + str(j) +')'
                filename = cwd+ '/models/' + Net_name
                filehandler = open(filename, 'rb') 
                Net = pickle.load(filehandler)
                filehandler.close()
                
                ELL, ELBO = plot_validation(Net, Net_name)
                update_log(Net, Net_name, ELL, ELBO)

                
