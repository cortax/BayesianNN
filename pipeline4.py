import torch
import math
import pandas as pd

from pandas import read_csv

import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch import tensor
from torch.utils.data import Dataset
from os import listdir
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from collections import OrderedDict

from matplotlib import pyplot as plt

# The training details are printed in this file
f_log = open("../DataBombardier/log.txt", "w")


inputs_idx = (0, 2, 3, 4, 5, 8) # The input columns used for training (from 0 to 11)
outputs_idx = 6 # The temperature column

def normalization_func(dfs):
    array = np.concatenate([df.to_numpy() for df in dfs], axis=0)
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    output = lambda df: (df - mean)/std
    return output

def denormalization_temp_func(dfs):
    array = np.concatenate([df.to_numpy() for df in dfs], axis=0)
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    output = lambda df: mean[outputs_idx] + (df*std[outputs_idx])
    return output


def load_dfs(path_name, columns=range(1,12)):

    assert(isinstance(path_name, str))

    if path_name[-1] == '/':
        path_name = path_name[:-1]

    file_names = listdir(path_name)

    dfs = []

    for file_name in file_names:
        if file_name.endswith('.csv'):
            df = read_csv(path_name + '/' + file_name, usecols=columns, skiprows=(1,2),index_col=0,dtype=np.float32)
            index_np = df.index.to_numpy()
            df = df.reindex(pd.Float64Index(data=np.arange(index_np[0],index_np[-1]+1.0,1)))
            df = df.set_index(pd.TimedeltaIndex(df.index,unit='s'))
            df = df.interpolate()
            dfs.append(df)

    return dfs


class DatasetBasic(Dataset):

    def __init__(self, dfs, past_steps, future_steps, stride, skip_ts, transform=lambda dfs:dfs):

        self.past_steps = past_steps
        self.future_steps = future_steps

        total_steps = past_steps + future_steps

        list_xs = []
        list_ys = []

        dfs = transform(dfs)

        for df in dfs:

            D = df.to_numpy()

            t0s = range(0, max([len(D) - skip_ts * total_steps, 1]), stride)

            for t0 in t0s:
                w = slice(t0,(t0 + skip_ts * total_steps),skip_ts)

                list_xs.append(tensor(D[w, inputs_idx]))
                list_ys.append(tensor(D[w, outputs_idx]))

        self.size_train = len(list_xs)

        self.xs = torch.stack(list_xs)
        self.ys = torch.stack(list_ys)

        return

    def __getitem__(self, index):

        X = (self.xs[index],self.ys[index,:self.past_steps])
        Y = self.ys[index,self.past_steps:]

        return (X,Y)

    def __len__(self):
        return self.size_train


class NNModelBasic(nn.Module):

    def __init__(self, dim_exo, dim_endo, past_steps, layer_width):

        super(NNModelBasic, self).__init__()
        withbias = True

        H = layer_width

        self.past_steps = past_steps

        self.dim_exo = dim_exo
        self.dim_endo = dim_endo

        len_inputs = (dim_exo * (past_steps + 1) + dim_endo * past_steps)

        # Takes as input the output variables (columns given by outputs_idx)
        self.ExoLayer = nn.Linear(dim_exo*(past_steps+1), H, bias=False)
        nn.init.uniform_(self.ExoLayer.weight, -math.sqrt(6./(len_inputs*H)), math.sqrt(6./(len_inputs*H)))

        # Takes as input the input variables (columns given by inputs_idx)
        self.EndoLayer = nn.Linear(dim_endo*past_steps, H, bias=withbias)
        nn.init.uniform_(self.EndoLayer.weight, -math.sqrt(6./(len_inputs*H)), math.sqrt(6./(len_inputs*H)))

        self.PostFullyConnected = nn.Sequential(OrderedDict([
            ('PostRelu0', nn.LeakyReLU(negative_slope=1e-2)),
            ('PostLayer1', nn.Linear(H, H, bias=withbias)), ('PostAct1', nn.LeakyReLU(negative_slope=1e-2)),
            ('PostLayerLast', nn.Linear(H, dim_endo, bias=withbias))
        ]))

        for module in self.PostFullyConnected.modules():
            if type(module) == nn.Linear:
                nn.init.uniform_(module.weight,-math.sqrt(6./(H*H)), math.sqrt(6./(H*H)))

    def forward(self, inputs):

        xs, ys = inputs

        future_steps = xs.shape[1] - ys.shape[1]

        for t in range(future_steps):

            h_exo = self.ExoLayer(xs.narrow(1,t,self.past_steps+1).view((xs.shape[0], (self.past_steps+1)*self.dim_exo)))
            h_endo = self.EndoLayer(ys.narrow(1,t,self.past_steps).view((ys.shape[0], self.past_steps*self.dim_endo)))

            h = h_exo + h_endo

            pred = self.PostFullyConnected(h).view((len(ys), self.dim_endo))
            ys = torch.cat((ys,pred),dim=1)

        return ys[:,-future_steps:]


def train(model, data_train, data_val, criterion, n_epoch, learning_rate, batch_size):
    use_gpu = True
    device = 1       # Id of the GPU used (Set to 0 if you don't have more than 1 GPU!)
    n_cpu = 10       # Number of CPUs available (used for loading data onto the GPU)

    train_loader = DataLoader(data_train, batch_size=batch_size,shuffle=True, num_workers=n_cpu)
    val_loader = DataLoader(data_val, batch_size=batch_size,shuffle=True, num_workers=n_cpu)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    disp_train_loss = lambda t, loss: print('Train (epoch ' + str(t) + '): ' + str(loss))
    disp_val_loss = lambda t, loss: print('Val (epoch ' + str(t) + '): ' + str(loss) + '\n')

    for t in range(n_epoch):
        model.eval()    # Do not compute the gradient
        model.cpu()     # Make all calculations on the cpu

        train_losses.append(train_loss(model, train_loader))
        val_losses.append(train_loss(model, val_loader))

        disp_train_loss(t, train_losses[-1])
        disp_val_loss(t, val_losses[-1])

        model.train()
        if use_gpu:
            model.cuda(device)

        for batch in train_loader:

            inputs , targets = batch
            if use_gpu:
                xs,ys = inputs
                xs = xs.cuda(device)
                ys = ys.cuda(device)
                inputs = (xs,ys)
                targets = targets.cuda(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs[:, -101:None:20], targets[:, -101:None:20])

            loss.backward()
            optimizer.step()

    model.eval()    # Do not compute the gradient
    model.cpu()     # Make all calculations on the cpu

    train_losses.append(train_loss(model, train_loader))
    val_losses.append(train_loss(model, val_loader))

    disp_train_loss(t+1, train_losses[-1])
    disp_val_loss(t+1, val_losses[-1])

    return train_losses, val_losses



def train_loss(model, train_loader):

    criterion_sum = nn.MSELoss(reduction='sum')

    model.eval()    # Do not compute the gradient
    model.cpu()     # Make all calculations on the cpu
    val_loss = []

    for batch in train_loader:
        inputs, targets = batch
        output = model(inputs)
        val_loss.append(criterion_sum(output,targets).item())

    return math.sqrt(np.sum(val_loss)/(len(train_loader.dataset)*train_loader.dataset.future_steps))


if __name__ == "__main__":

    # These 4 lines make the whole code deterministic
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    # Number of NARX in the ensemble
    ensemble_size = 1

    past_steps = 32     # 32 by default (aka causal span)
    future_steps = 80   # 80 by default  (aka effex span)
    stride = 60         # 1 out of "stride" possible examples is kept for training
    skip_ts = 60        # A jump of "skip_ts" time steps is made in each point of an example

    # Folder where the csv are
    datafolder = '../DataBombardier/'

    layer_width = 100       # Width of the hidden layers
    n_epoch = 30            # Fixed number of epochs
    batch_size = 32         # Size of one batch (one gradient step for each batch)
    learning_rate = 1e-4

    # All the data frames
    dfs = load_dfs(datafolder)

    transform = normalization_func(dfs[1:])         # Normalization lambda function
    untransform = denormalization_temp_func(dfs)    # Unnormalization lambda function

    df_test = [transform(df) for df in dfs[0:1]]
    dfs_train = [transform(df) for df in dfs[1:]]

    data_train = DatasetBasic(dfs_train, past_steps, future_steps, stride, skip_ts)
    data_test = DatasetBasic(df_test, past_steps, future_steps, stride, skip_ts)

    datas_whole = [DatasetBasic([transform(df)], past_steps, 1000000, stride, skip_ts) for df in dfs]

    models = []
    for n in range(ensemble_size):
        models.append(NNModelBasic(dim_endo=1, dim_exo=len(inputs_idx), past_steps=past_steps, layer_width=layer_width))

    n_params = sum(parameters.numel() for parameters in models[0].parameters())

    print('Past steps : %d\nFuture steps : %d\n' %(past_steps, future_steps), file=f_log)
    print('Stride : %d\nskip ts : %d\n' %(stride, skip_ts), file=f_log)
    print('LayerWidth: %d\nn_epoch: %d\n' %(layer_width, n_epoch), file=f_log)
    print('BatchSize : %d\nLearningRate : %f\n' %(batch_size, learning_rate), file=f_log)
    print('Size (train) : ' + str(len(data_train)), file=f_log)
    print('Size (test) : ' + str(len(data_test)), file=f_log)
    print('# of parameters : ' + str(n_params), file=f_log)

    criterion = nn.MSELoss()

    train_losses = []   # Contains the learning curves for the training set
    test_losses = []    # Contains the learning curves for the test set

    for n in range(ensemble_size):
        print('NARX ' + str(n+1) + '(Training)')
        train_loss, test_loss = train(models[n], data_train, data_test, criterion, n_epoch=n_epoch, batch_size=batch_size, learning_rate=learning_rate)
        models[n].cpu()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plt.semilogy(train_losses,label='Train (flights 2,..,13')
    plt.semilogy(test_losses, label='Test (flight 1)')
    plt.xlabel('epoch')
    plt.ylabel('RMSE (normalized)')
    plt.savefig('../DataBombardier/loss.png')
    plt.cla()

    inputs_train, targets_train = data_train[0:len(data_train)]
    inputs_test, targets_test = data_test[0:len(data_test)]

    for n,data_whole in enumerate(datas_whole):

        inputs_test_whole, targets_test_whole = data_whole[0:len(data_whole)]

        for m in range(ensemble_size):
            outputs_whole = models[m](inputs_test_whole)
            temperatures_pred = untransform(outputs_whole.detach().numpy().flatten())
            plt.plot(range(past_steps,past_steps+len(temperatures_pred)),temperatures_pred, color='red', label='NARX' if m==0 else None)

        temperatures_ground = dfs[n].to_numpy()[0::skip_ts,outputs_idx]

        plt.plot(temperatures_ground, color='blue', label='Data')
        plt.xlabel('Time steps (Delta = ' + str(skip_ts) + 's)')
        plt.ylabel('Temperature (C)')
        plt.title('Flight #' + str(n+1))
        plt.legend(loc='lower left')
        plt.savefig('../DataBombardier/flight' + str(n+1) + '.png')
        plt.cla()

        print('Test Whole flight (#' + str(n+1) + ') :' + str(torch.sqrt(criterion(outputs_whole, targets_test_whole)).item()),file=f_log)

    for n in range(ensemble_size):
        print('Train RMSE (with normalization) (NARX #' + str(n+1) + '): ' + str(torch.sqrt(criterion(untransform(models[n](inputs_train)), targets_train)).item()) + '\n', file=f_log)
        print('Test RMSE (with normalization): (NARX #' + str(n+1) + '): ' + str(torch.sqrt(criterion(untransform(models[n](inputs_test)), targets_test)).item()) + '\n', file=f_log)

f_log.close()
