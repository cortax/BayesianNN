import torch
import math
import pandas as pd

from pandas import read_csv
from pandas import DataFrame

import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch import tensor
from torch.utils.data import Dataset
from os import listdir
from torch.utils.data import DataLoader
from collections import OrderedDict

from matplotlib import pyplot as plt

f_log = open("../DataBombardier/log.txt", "w")
inputs_idx = (0, 2, 3, 4, 5, 8)
outputs_idx = (6)

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

        dims = (dim_exo * (past_steps + 1) + dim_endo * past_steps) * H

        self.ExoLayer = nn.Linear(dim_exo*(past_steps+1), H, bias=False)
        nn.init.uniform_(self.ExoLayer.weight, -math.sqrt(6./dims), math.sqrt(6./dims))

        self.EndoLayer = nn.Linear(dim_endo*past_steps, H, bias=withbias)
        nn.init.uniform_(self.EndoLayer.weight, -math.sqrt(6./dims), math.sqrt(6./dims))

        self.PostFullyConnected = nn.Sequential(OrderedDict([
            ('PostRelu0', nn.LeakyReLU(negative_slope=1e-2)),
            ('PostLayer1', nn.Linear(H, H, bias=withbias)), ('PostAct1', nn.LeakyReLU(negative_slope=1e-2)),
            ('DropoutLayer', nn.Dropout()),
            ('PostLayerLast', nn.Linear(H, dim_endo, bias=withbias))
        ]))

        for module in self.PostFullyConnected.modules():
            if type(module) == nn.Linear:
                nn.init.uniform_(module.weight,-math.sqrt(6./dims), math.sqrt(6./dims))


    def forward(self, inputs):

        xs,ys = inputs

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
    device = 1

    train_loader = DataLoader(data_train, batch_size=batch_size,shuffle=True, num_workers=10)
    val_loader = DataLoader(data_val, batch_size=batch_size,shuffle=True, num_workers=10)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = MultiStepLR(optimizer, milestones=[10,50,100,200,300,400,500,600,700,800,900],gamma=0.5)

    train_losses = []
    val_losses = []

    disp_train_loss = lambda t, loss: print('Train (epoch ' + str(t) + '): ' + str(loss))
    disp_val_loss = lambda t, loss: print('Val (epoch ' + str(t) + '): ' + str(loss) + '\n')

    for t in range(n_epoch):
        model.eval()
        model.cpu()

        train_losses.append(train_loss(model, train_loader))
        val_losses.append(train_loss(model, val_loader))

        disp_train_loss(t, train_losses[-1])
        disp_val_loss(t, val_losses[-1])

        #scheduler.step(t)

        model.train()
        if use_gpu:
            model.cuda(device)

        #(scheduler.get_lr())
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

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

    model.eval()
    model.cpu()

    train_losses.append(train_loss(model, train_loader))
    val_losses.append(train_loss(model, val_loader))

    disp_train_loss(t+1, train_losses[-1])
    disp_val_loss(t+1, val_losses[-1])

    return train_losses, val_losses


def train_loss(model, train_loader):

    criterion = nn.MSELoss(reduction='sum')

    model.eval()
    model.cpu()
    val_loss = []

    for batch in train_loader:
        inputs, targets = batch
        output = model(inputs)
        val_loss.append(criterion(output,targets).item())

    return math.sqrt(np.sum(val_loss)/(len(train_loader.dataset)*train_loader.dataset.future_steps))




if __name__ == "__main__":

    #torch.manual_seed(42)

    past_steps = 32     # 32 by default
    future_steps = 80  # 80 by default
    stride = 5
    skip_ts = 60

    datafolder = '../DataBombardier/'

    layer_width = 100
    n_epoch = 10
    batch_size = 256
    learning_rate = 1e-4

    dfs = load_dfs(datafolder)
    transform = normalization_func(dfs[1:])
    untransform = denormalization_temp_func(dfs)

    df_test = [transform(df) for df in dfs[0:1]]
    dfs_train = [transform(df) for df in dfs[1:]]

    data_train = DatasetBasic(dfs_train, past_steps, future_steps, stride, skip_ts)
    data_test = DatasetBasic(df_test, past_steps, future_steps, stride, skip_ts)

    datas_whole = [DatasetBasic([transform(df)], past_steps, 1000000, stride, skip_ts) for df in dfs]

    model = NNModelBasic(dim_endo=1, dim_exo=len(inputs_idx), past_steps=past_steps,layer_width=layer_width)

    n_params = sum(parameters.numel() for parameters in model.parameters())

    print('Past steps : %d\nFuture steps : %d\n' %(past_steps, future_steps), file=f_log)
    print('Stride : %d\nskip ts : %d\n' %(stride, skip_ts), file=f_log)
    print('LayerWidth: %d\nn_epoch: %d\n' %(layer_width, n_epoch), file=f_log)
    print('BatchSize : %d\nLearningRate : %f\n' %(batch_size, learning_rate), file=f_log)
    print('Size (train) : ' + str(len(data_train)), file=f_log)
    print('Size (test) : ' + str(len(data_test)), file=f_log)
    print('# of parameters : ' + str(n_params), file=f_log)

    criterion = nn.MSELoss()
    #criterion_all = lambda outputs_all, targets_all: \
    #    criterion(outputs,targets)

    train_losses, test_losses = train(model, data_train, data_test, criterion, n_epoch=n_epoch, batch_size=batch_size, learning_rate=learning_rate)
    model.cpu()

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

        outputs_whole = model(inputs_test_whole)
        temperatures_pred = untransform(outputs_whole.detach().numpy().flatten())
        temperatures_ground = untransform(targets_test_whole.detach().numpy().flatten())

        plt.plot(temperatures_ground, color='blue', label='Data')
        plt.plot(temperatures_pred, color='red', label='NARX')
        plt.xlabel('Time steps (Delta = 60s)')
        plt.ylabel('Temperature (C)')
        plt.title('Flight #' + str(n + 1))
        plt.legend()
        plt.savefig('../DataBombardier/flight' + str(n+1) + '.png')
        plt.cla()

        print('Test Whole flight (#' + str(n+1) + ') :' + str(torch.sqrt(criterion(outputs_whole, targets_test_whole)).item()),file=f_log)

    print('Train RMSE (with normalization): ' + str(torch.sqrt(criterion(model(inputs_train),targets_train)).item()),file=f_log)
    print('Test RMSE (with normalization): ' + str(torch.sqrt(criterion(model(inputs_test), targets_test)).item()),file=f_log)
    torch.save(model.state_dict(), "../DataBombardier/model1")

    f_log.close()
