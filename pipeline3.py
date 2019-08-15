import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from collections import OrderedDict
from dataprocessing import load_dfs
from dataprocessing import DatasetBasic
from matplotlib import pyplot as plt

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
            ('PostLayerLast', nn.Linear(H, 1, bias=withbias))
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
            pred = ys[:,(self.past_steps-1+t,)] + self.PostFullyConnected(h).view((len(ys), 1))
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

    past_steps = 16     # 32 by default
    future_steps = 200  # 80 by default
    stride = 50
    skip_ts = 2

    print('Past steps : %d\nFuture steps : %d\nStride : %d\nskip ts : %d\n' %(past_steps, future_steps, stride, skip_ts))

    datafolder = '../DataBombardier/'

    layer_width = 30
    n_epoch = 30
    batch_size = 256
    learning_rate = 0.000001

    print('LayerWidth: %d\nn_epoch: %d\nBatchSize : %d\nLearningRate : %d\n' %(layer_width, n_epoch, batch_size, learning_rate))

    dfs = load_dfs(datafolder)
    df_test = dfs[0]
    dfs_train = dfs[1:]

    data_train = DatasetBasic(dfs_train, past_steps, future_steps, stride, skip_ts,True)
    data_test = DatasetBasic([df_test], past_steps, future_steps, stride, skip_ts, True)

    datas_whole = [DatasetBasic([df], past_steps, 1000000, stride, skip_ts, False) for df in dfs]

    model = NNModelBasic(dim_endo=1, dim_exo=6, past_steps=past_steps,layer_width=layer_width)

    n_params = sum(parameters.numel() for parameters in model.parameters())
    print('# of parameters : ' + str(n_params))

    print('Size (train) : ' + str(len(data_train)))
    print('Size (test) : ' + str(len(data_test)))

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

        print('Test Whole flight (#' + str(n) + ') :' + str(torch.sqrt(criterion(outputs_whole, targets_test_whole)).item()))

    print('Train RMSE: ' + str(torch.sqrt(criterion(model(inputs_train),targets_train)).item()))
    print('Test RMSE : ' + str(torch.sqrt(criterion(model(inputs_test), targets_test)).item()))
    torch.save(model.state_dict(), "../DataBombardier/model1")
