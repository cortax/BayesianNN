import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from dataprocessing import load_dfs, dataset_df_to_tensor


class NNModel(nn.Module):
    def __init__(self, input_dim, H):
        super(NNModel, self).__init__()

        self.linear1 = nn.Linear(input_dim, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, H)
        self.fc = nn.Linear(H, 1)

    def forward(self, x):
        h_relu1 = F.relu(self.linear1(x))
        h_relu2 = F.relu(self.linear2(h_relu1))
        h_relu3 = F.relu(self.linear3(h_relu2))
        y = self.fc(h_relu3)
        return y


def train_bptt(model, flighttests, n_epoch=2, learning_rate=0.01, batch_size=32):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for n in range(n_epoch):

        input = torch.Tensor(flighttests)
        target = torch.Tensor(flighttests)

        optimizer.zero_grad()

        output = model(input)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()


    return


def train(model, X,y, n_epoch=2, learning_rate=0.01, batch_size=32):

    trainloader = DataLoader(TensorDataset(X,y), batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    disp_loss = lambda t: print('Train (epoch ' + str(t) + '): ' + str(torch.sqrt(criterion(model(X), y)).item()))

    for t in range(n_epoch):
        disp_loss(t)

        for batch in trainloader:

            inputs, targets = batch
            optimizer.zero_grad()

            print(inputs.shape)
            print(targets.shape)

            output = model(inputs)

            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()

    disp_loss(t)
    return


if __name__ == "__main__":

    K = 120
    with_y = True

    datafolder = '../DataBombardier/'

    tau = 1
    stride = 30
    inputs_idx = (1,2,3,4,5)
    skip_row = 1

    dfs = load_dfs(datafolder)
    df_test = dfs[0]
    dfs_train = dfs[1:]

    Xs = []; ys = []
    for df in dfs_train:

        X,y = dataset_df_to_tensor(df, K, stride=stride, skip_ts=skip_row, inputs_idx=inputs_idx, tau=tau, with_y=with_y)

        Xs.append(X)
        ys.append(y)

    X_train = torch.cat(Xs[1:],0)
    y_train = torch.cat(ys[1:],0)

    X_test = Xs[0]
    y_test = ys[0]

    model = NNModel(X_train.shape[1], 1)

    print('Size (train) : ' + str(X_train.shape[0]) + ' x ' + str(X_train.shape[1]))

    train(model, X_train, y_train, n_epoch=300, batch_size=len(X))
    criterion = nn.L1Loss()

    print('Train: ' + str(criterion(model(X_train), y_train).item()))
    print('Test : ' + str(criterion(model(X_test),y_test).item()))
