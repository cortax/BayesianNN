import sys
import os
from os.path import dirname

try:
    rootdir = os.path.dirname(os.path.realpath(__file__))
except:
    rootdir = os.getcwd()

rootdir = rootdir.split('BayesianNN')[0]+'BayesianNN/'
print(rootdir)
sys.path.append( rootdir )

cwd = rootdir + 'Experiments/foong/tanh/MAP+BBVI/'
print(cwd)

from Inference import BBVI 
import _pickle as pickle
import torch
import time 

def train_model(layer_width, nb_layers, activation, seed, Net):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    Net.set_device(device)
    #Net.requires_grad_mus(False)
    
    data = torch.load(rootdir + 'Data/foong_data.pt')
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = y_data.unsqueeze(-1)
    
    print(Net.compute_elbo(x_data, y_data, n_samples_ELBO=1000, sigma_noise=0.1, device=device).detach().cpu().numpy())

    optimizer = torch.optim.Adam
    optimizer_params = {'lr': 0.1}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {'patience': 2, 'factor': 0.8, 'threshold': 1e-3, 'threshold_mode': 'abs'}

    voptimizer = BBVI.VariationalOptimizer(model=Net, sigma_noise=0.1, optimizer=optimizer, optimizer_params=optimizer_params, scheduler=scheduler, scheduler_params=scheduler_params, min_lr=0.00001)
    Net = voptimizer.run((x_data,y_data), n_epoch=100000, n_iter=150, seed=seed, n_ELBO_samples=75, verbose=1)

    training_infos = [str(voptimizer.last_epoch), str(optimizer), str(optimizer_params), str(scheduler), str(scheduler_params)] 

    return Net, training_infos

if __name__ == "__main__":
    activation = torch.tanh

    print('making dirs')
    os.makedirs(os.path.dirname(cwd+'models/'), exist_ok=True) 
    os.makedirs(os.path.dirname(cwd+'logs/'), exist_ok=True) 

    with open('job_parameters_array', 'r') as f:
        lines = f.read().splitlines()
        
    idx = int(sys.argv[1])
    
    args = lines[idx].split(';')
    L = int(args[0])
    W = int(args[1])
    j = int(args[2])
    

    filename = str(L)+ 'Layers_' + str(W) + 'Neurons_(' + str(j) +')'
    pathname = cwd+'models/'

    print(filename)
    print(pathname)

    if not os.path.exists(pathname+filename): 
        filehandler = open(rootdir + 'Experiments/foong/tanh/MAP/models/' + filename, 'rb')
        netparam = pickle.load(filehandler)
        Net = BBVI.VariationalNetwork(input_size=netparam['input_size'],
                              output_size=netparam['output_size'],
                              layer_width=netparam['layer_width'],
                              nb_layers=netparam['nb_layers'])
        Net.set_network(netparam)
        
        start_time = time.time() 
        Net, training_infos = train_model(W, L, activation, j, Net)
        training_time = time.time() - start_time 

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data = torch.load(rootdir + 'Data/foong_data.pt')
        x_data = data[0].to(device)
        y_data = data[1].to(device)
        y_data = y_data.unsqueeze(-1)

        
        print(Net.compute_elbo(x_data, y_data, n_samples_ELBO=1000, sigma_noise=0.1, device=device).detach().cpu().numpy())

        filehandler = open(pathname+filename, 'wb') 
        netparam = Net.get_network()
        pickle.dump(netparam, filehandler)
        filehandler.close()

        log = open(cwd + 'logs/' + filename + '.txt', 'w+')
        log.write('Training time: ' + str(training_time) + '\n') 
        log.write('Number of epochs: ' + training_infos[0] + '\n') 

        log.write('Optimizer: ' + training_infos[1] + ', ' + training_infos[2] + '\n') 
        log.write('Scheduler: ' + training_infos[3] + ', ' + training_infos[4] + '\n') 
        log.close()        
