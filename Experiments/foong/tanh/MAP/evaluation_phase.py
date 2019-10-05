import sys
import os
from os.path import dirname
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dirname(dirname(dirname(dirname(cwd)))) )

from Inference import BBVI 
import _pickle as pickle
import torch

def evaluate_model(layer_width, nb_layers, activation, seed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.load('data/foong_data_validation.pt')
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = y_data.unsqueeze(-1)


if __name__ == "__main__":
    activation = torch.tanh
    layer_size = [10,25,50,100]
    nb_layer = [2,3,4]
    nb_trial = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.load('data/foong_data_validation.pt')
    x_data_validation = data[0].to(device)
    y_data_validation = data[1].to(device)
    y_data_validation = y_data_validation.unsqueeze(-1)

    


    for L in nb_layer:
        for W in layer_size:
            for j in range(nb_trial):
                filename = cwd+'/models/' + str(L)+ 'Layers_' + str(W) + 'Neurons_(' + str(j) +')'
                filehandler = open(filename, 'rb') 
                Net = pickle.load(filehandler)
                filehandler.close()

                y_pred_validation = Net.forward(x_data_validation)
                print(y_pred_validation)
    