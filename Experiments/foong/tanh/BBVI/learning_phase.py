import sys
import os
from os.path import dirname
cwd = os.path.dirname(os.path.realpath(__file__))
rootdir = dirname(dirname(dirname(dirname(cwd))))
sys.path.append( rootdir )

from Inference import BBVI 
import _pickle as pickle
import torch
import FTPTools

def train_model(layer_width, nb_layers, activation, seed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.load(rootdir + '/Data/foong_data.pt')
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = y_data.unsqueeze(-1)

    optimizer = torch.optim.Adam
    optimizer_params = {'lr': 0.1}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {'patience': 5, 'factor': 0.8}

    Net = BBVI.VariationalNetwork(input_size=1, output_size=1, layer_width=layer_width, nb_layers=nb_layers, activation=activation, device=device)

    voptimizer = BBVI.VariationalOptimizer(model=Net, sigma_noise=0.1, optimizer=optimizer, optimizer_params=optimizer_params, scheduler=scheduler, scheduler_params=scheduler_params, min_lr=0.00001)
    Net = voptimizer.run((x_data,y_data), n_epoch=100000, n_iter=150, seed=seed, n_ELBO_samples=75, verbose=1)

    training_infos = [str(optimizer), str(optimizer_params), str(scheduler), str(scheduler_params)]

    return Net, training_infos

if __name__ == "__main__":
    activation = torch.tanh
    layer_size = [10,25,50,100]
    nb_layer = [2,3,4]
    nb_trial = 10

    os.makedirs(os.path.dirname(cwd+'/models/'), exist_ok=True) 
    os.makedirs(os.path.dirname(cwd+'/logs/'), exist_ok=True) 

    for L in nb_layer:
        for W in layer_size:
            for j in range(nb_trial):
                filename = str(L)+ 'Layers_' + str(W) + 'Neurons_(' + str(j) +')'
                pathname = cwd+'/models/'

                if not FTPTools.fileexists(pathname.split('Experiments')[1], filename):
                    Net, training_infos = train_model(W, L, activation, j)

                    filehandler = open(pathname+filename, 'wb') 
                    pickle.dump(Net, filehandler)
                    filehandler.close()

                    filehandler = open(pathname+filename, 'rb') 
                    FTPTools.upload(filehandler, pathname.split('Experiments')[1], filename)
                    filehandler.close()

                    log = open(cwd + '/logs/' + filename + '.txt', 'w+')
                    log.write('Optimizer: ' + training_infos[0] + ', ' + training_infos[1] + '\n')
                    log.write('Scheduler: ' + training_infos[2] + ', ' + training_infos[3] + '\n')
                    log.close()

                    filehandler = open(cwd + '/logs/' + filename + '.txt', 'rb') 
                    logpathname = cwd+'/logs/'
                    FTPTools.upload(filehandler, logpathname.split('Experiments')[1], filename + '.txt')
                    filehandler.close()