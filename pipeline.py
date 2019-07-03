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
        #h_relu3 = F.relu(self.linear3(h_relu2))
        y = self.fc(h_relu2)
        return y


def validate(model, trainloader):
    return None


def train(model, X,y, n_epoch=2, learning_rate=0.01):

    trainloader = DataLoader(TensorDataset(X,y), batch_size=2056)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    disp_loss = lambda t: print('Train (epoch ' + str(t) + '): ' + str(torch.sqrt(criterion(model(X), y)).item()))

    for t in range(n_epoch):
        disp_loss(t)

        for batch in trainloader:

            inputs, targets = batch
            optimizer.zero_grad()

            output = model(inputs)

            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()

    disp_loss(t)



if __name__ == "__main__":

    dfs = load_dfs("../DataBombardier/")
    K = 10
    Xs = []; ys = []

    df_test = dfs[0]
    dfs_train = dfs[1:]

    X_test,y_test = dataset_df_to_tensor(df_test, K)

    for df in dfs_train:
        X,y = dataset_df_to_tensor(df, K)
        Xs.append(X)
        ys.append(y)

    X_train = torch.cat(Xs,0)
    y_train = torch.cat(ys,0)

    print(X_train.shape)
    print(X_test.shape)

    model = NNModel(X_train.shape[1], 100)
    train(model, X_train, y_train, n_epoch=5)

    criterion = nn.L1Loss()

    print('Train: ' + str(criterion(model(X_train), y_train).item()))
    print('Test : ' + str(criterion(model(X_test),y_test).item()))


