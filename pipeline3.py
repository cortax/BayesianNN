import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from collections import OrderedDict

from dataprocessing import load_dfs
from dataprocessing import DatasetBasic

class NNModelBasic(nn.Module):

    def __init__(self, dim_exo, dim_endo, past_steps, future_steps, layer_width):

        super(NNModelBasic, self).__init__()

        H = layer_width

        self.past_steps = past_steps
        self.future_steps = future_steps

        self.ExoLayers = nn.ModuleList([nn.Linear(dim_exo,H, bias=False) for n in range(past_steps + 1)])
        self.EndoLayers = nn.ModuleList([nn.Linear(dim_endo,H, bias=(n==0)) for n in range(past_steps)])

        self.PostFullyConnected = nn.Sequential(OrderedDict([
            ('PostRelu0', nn.ReLU()),
            ('PostLayer1', nn.Linear(H, H)), ('PostRelu1', nn.ReLU()),
            ('PostLayer2', nn.Linear(H, H)), ('PostRelu2', nn.ReLU()),
            ('PostLayer3', nn.Linear(H, H)), ('PostRelu3', nn.ReLU()),
            ('PostLayer4', nn.Linear(H, H)), ('PostRelu4', nn.ReLU()),
            ('PostLayer5', nn.Linear(H, H)), ('PostRelu5', nn.ReLU()),
            ('PostLayer6', nn.Linear(H, H)), ('PostRelu6', nn.ReLU()),
            ('PostLayer7', nn.Linear(H, H)), ('PostRelu7', nn.ReLU()),
            ('PostLayerLast', nn.Linear(H, 1))
        ]))

    def forward(self, inputs):

        xs_all = list(inputs[0])
        ys_all = list(inputs[1])

        outputs = []

        for t in range(self.future_steps):
            xs = xs_all[t:(self.past_steps+t+1)]
            ys = ys_all[t:(self.past_steps+t)]

            h_exo = sum([self.ExoLayers[n](xs[n]) for n in range(self.past_steps + 1)])
            h_endo = sum([self.EndoLayers[n](ys[n]) for n in range(self.past_steps)])

            h = h_exo + h_endo
            ys_all.append(self.PostFullyConnected(h))

        return ys_all[-future_steps:]



def train(model, data,criterion, n_epoch, learning_rate, batch_size):

    use_gpu = True
    device = 1

    inputs_all, targets_all = data[range(0, len(data))]
    trainloader = DataLoader(data, batch_size=batch_size,shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    disp_loss = lambda t: print('Train (epoch ' + str(t) + '): ' + str(torch.sqrt(criterion(model(inputs_all), targets_all)).item()))

    for t in range(n_epoch):
        model.cpu()
        disp_loss(t)

        if use_gpu:
            model.cuda(device)

        for batch in trainloader:

            inputs , targets = batch
            if use_gpu:
                xs,ys = inputs
                xs = tuple(x.cuda(device) for x in xs)
                ys = tuple(y.cuda(device) for y in ys)
                inputs = (xs,ys)
                targets = [target.cuda(device) for target in targets]

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

    model.cpu()
    disp_loss(t)
    return


if __name__ == "__main__":

    #torch.manual_seed(42)

    n_epoch = 5

    past_steps = 10
    future_steps = 10
    with_y = True

    datafolder = '../DataBombardier/'

    stride = 10
    skip_ts = 10
    n_epoch = 500
    batch_size = 1000

    inputs_idx = (1,2,3,4,5)

    dfs = load_dfs(datafolder)
    df_test = dfs[0]
    dfs_train = dfs[1:]

    data_train = DatasetBasic(dfs_train, past_steps, future_steps, stride, skip_ts)
    data_test = DatasetBasic([df_test], past_steps, future_steps, stride, skip_ts)

    model = NNModelBasic(dim_endo=1, dim_exo=5, past_steps=past_steps,future_steps=future_steps, layer_width=6)

    n_params = sum(parameters.numel() for parameters in model.parameters())
    print('# of parameters : ' + str(n_params))

    n_train = len(data_train)
    n_test = len(data_test)

    print('Size (train) : ' + str(n_train))
    print('Size (test) : ' + str(len(data_test)))

    criterion = nn.MSELoss()
    criterion_all = lambda outputs_all, targets_all: \
        sum([criterion(outputs,targets) for (outputs,targets) in zip(outputs_all,targets_all)])

    train(model, data_train,criterion_all, n_epoch=n_epoch, batch_size=batch_size, learning_rate=0.01)

    inputs_train, outputs_train = data_train[0:n_train]
    inputs_test, outputs_test = data_test[0:n_test]

    print('Train RMSE: ' + str(torch.sqrt(criterion_all(model(inputs_train),outputs_train)).item()))
    print('Test RMSE:  ' + str(torch.sqrt(criterion_all(model(inputs_test), outputs_test)).item()))

    y0 = data_test.ys[0]

    print('Test Baseline : ' + str(float(torch.sqrt(criterion(y0[:-1], y0[1:])))))
    torch.save(model.state_dict(), "../DataBombardier/model1")
