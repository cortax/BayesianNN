import torch
import math
import pandas as pd

from pandas import read_csv
from pandas import DataFrame

import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset
from os import listdir
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from collections import OrderedDict

from matplotlib import pyplot as plt

f_log = open("log.txt", "w")
inputs_idx = (0, 2, 3, 4, 5, 8)
outputs_idx = (6)

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    #s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), x, mode='valid')
    return y


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

    def __init__(self, dfs, past_steps, future_steps, stride, skip_ts, smoothing):

        self.past_steps = past_steps
        self.future_steps = future_steps


        total_steps = past_steps + future_steps

        list_xs = []
        list_ys = []

        for df in dfs:

            D = df.to_numpy()

            if smoothing:
                D[499:-500,outputs_idx] = smooth(D[:,outputs_idx], 1000)
                t0s = range(499, max([len(D) - 500 - stride * total_steps, 1]), skip_ts)
            else:
                t0s = range(0, max([len(D) - stride * total_steps, 1]), skip_ts)

            for t0 in t0s:
                list_xs.append(tensor(D[t0:(t0+stride*total_steps):stride, inputs_idx]))
                list_ys.append(tensor(D[t0:(t0+stride*total_steps):stride, outputs_idx]))

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

        dims = dim_exo * (past_steps + 1) + dim_endo * past_steps

        self.ExoLayer = nn.Linear(dim_exo*(past_steps+1), H, bias=False)
        nn.init.normal_(self.ExoLayer.weight, mean=0.0, std=0.1/dims)

        self.EndoLayer = nn.Linear(dim_endo*past_steps, H, bias=withbias)
        nn.init.normal_(self.EndoLayer.weight, mean=0.0, std=0.1/dims)

        self.PostFullyConnected = nn.Sequential(OrderedDict([
            ('PostRelu0', nn.Tanh()),
            ('PostLayer1', nn.Linear(H, H, bias=withbias)), ('PostAct1', nn.Tanh()),
            ('PostLayer2', nn.Linear(H, H, bias=withbias)), ('PostAct2', nn.Tanh()),
            #('PostLayer3', nn.Linear(H, H, bias=withbias)), ('PostAct3', nn.Tanh()),
            #('PostLayer4', nn.Linear(H, H, bias=withbias)), ('PostAct4', nn.Tanh()),
            #('PostLayer5', nn.Linear(H, H, bias=withbias)), ('PostAct5', nn.Tanh()),
            #('PostLayer6', nn.Linear(H, H, bias=withbias)), ('PostAct6', nn.Tanh()),
            #('PostLayer7', nn.Linear(H, H, bias=withbias)), ('PostAct7', nn.Tanh()),
            ('PostLayerLast', nn.Linear(H, dim_endo, bias=withbias))
        ]))

        #for module in self.PostFullyConnected.modules():
            #if type(module) == nn.Linear:
                #nn.init.normal_(module.weight,mean=0.0,std=0.001/H)

    def forward(self, inputs):

        xs,ys = inputs

        future_steps = xs.shape[1] - ys.shape[1]

        for t in range(future_steps):

            h_exo = self.ExoLayer(xs.narrow(1,t,self.past_steps+1).view((xs.shape[0], (self.past_steps+1)*self.dim_exo)))
            h_endo = self.EndoLayer(ys.narrow(1,t,self.past_steps).view((ys.shape[0], self.past_steps*self.dim_endo)))

            h = h_exo + h_endo
            pred = ys[:,(self.past_steps-1+t,)] + self.PostFullyConnected(h).view((len(ys), self.dim_endo))
            ys = torch.cat((ys,pred),dim=1)

        return ys[:,-future_steps:]


def train(model, data_train, data_val, criterion, n_epoch, learning_rate, batch_size):

    use_gpu = True
    device = 1

    train_loader = DataLoader(data_train, batch_size=batch_size,shuffle=True, num_workers=10)
    val_loader = DataLoader(data_val, batch_size=batch_size,shuffle=True, num_workers=10)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[10,50,100,200,300,400,500,600,700,800,900],gamma=0.5)

    train_losses = []
    val_losses = []

    disp_train_loss = lambda t, loss: print('Train (epoch ' + str(t) + '): ' + str(loss), file=f_log)
    disp_val_loss = lambda t, loss: print('Val (epoch ' + str(t) + '): ' + str(loss) + '\n', file=f_log)

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

    past_steps = 16     # 32 by default
    future_steps = 200  # 80 by default
    stride = 50
    skip_ts = 2

    datafolder = '../DataBombardier/'

    layer_width = 60
    n_epoch = 30
    batch_size = 256
    learning_rate = 0.000001

    dfs = load_dfs(datafolder)
    df_test = dfs[0]
    dfs_train = dfs[1:]

    data_train = DatasetBasic(dfs_train, past_steps, future_steps, stride, skip_ts,True)
    data_test = DatasetBasic([df_test], past_steps, future_steps, stride, skip_ts, True)

    datas_whole = [DatasetBasic([df], past_steps, 1000000, stride, skip_ts, False) for df in dfs]

    model = NNModelBasic(dim_endo=1, dim_exo=6, past_steps=past_steps,layer_width=layer_width)

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

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.savefig('../DataBombardier/loss.png')
    plt.cla()

    inputs_train, targets_train = data_train[0:len(data_train)]
    inputs_test, targets_test = data_test[0:len(data_test)]

    for n,data_whole in enumerate(datas_whole):

        inputs_test_whole, targets_test_whole = data_whole[0:len(data_whole)]

        outputs_whole = model(inputs_test_whole)
        temperatures_pred = outputs_whole.detach().numpy().flatten()
        temperatures_ground = targets_test_whole.detach().numpy().flatten()

        plt.plot(temperatures_ground, color='blue')
        plt.plot(temperatures_pred, color='red')
        plt.savefig('../DataBombardier/flight' + str(n) + '.png')
        plt.cla()

        print('Test Whole flight (#' + str(n) + ') :' + str(torch.sqrt(criterion(outputs_whole, targets_test_whole)).item()),file=f_log)

    print('Train RMSE: ' + str(torch.sqrt(criterion(model(inputs_train),targets_train)).item()),file=f_log)
    print('Test RMSE : ' + str(torch.sqrt(criterion(model(inputs_test), targets_test)).item()),file=f_log)
    torch.save(model.state_dict(), "../DataBombardier/model1")

    f_log.close()
