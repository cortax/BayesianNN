import torch

import pandas as pd
import numpy as np

from pandas import read_csv
from pandas import DataFrame
from os import listdir

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import tensor

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DatasetBasic(Dataset):

    def __init__(self, dfs, past_steps, future_steps, stride, skip_ts):

        self.past_steps = past_steps
        self.future_steps = future_steps

        inputs_idx = (1,2,3,4,5)
        outputs_idx = (7,)

        total_steps = past_steps + future_steps

        list_xs = [[] for n in range(total_steps)]
        list_ys = [[] for n in range(total_steps)]

        for df in dfs:

            D = df.to_numpy()

            t0s = range(0, len(D) - stride*total_steps, skip_ts)

            for n in range(total_steps):
                for t0 in t0s:
                    list_xs[n].append(tensor(D[t0+stride*n, inputs_idx].flatten()))
                    list_ys[n].append(tensor(D[t0+stride*n, outputs_idx].flatten()))

        self.size_train = len(list_ys[0])

        self.xs = tuple(torch.stack(list_xs[n]) for n in range(total_steps))
        self.ys = tuple(torch.stack(list_ys[n]) for n in range(total_steps))

        return

    def __getitem__(self, index):

        X = (tuple(x[index] for x in self.xs),tuple(y[index] for y in self.ys[:self.past_steps]))
        Y = tuple(y[index] for y in self.ys[-self.future_steps:])

        return (X,Y)

    def __len__(self):
        return self.size_train



def load_dfs(path_name, columns=range(1,12)):

    assert(isinstance(path_name, str))

    if path_name[-1] == '/':
        path_name = path_name[:-1]

    file_names = listdir(path_name)

    dfs = []

    for file_name in file_names:
        if file_name.endswith('.csv'):
            df = read_csv(path_name + '/' + file_name, usecols=columns, skiprows=(1,2),index_col=0,dtype=np.float32)
            df = df.set_index(pd.TimedeltaIndex(df.index,unit='s'))
            dfs.append(df)

    return dfs



def dataset_df_to_tensor(df, K, skip_ts=5, stride=1, inputs_idx=(1,2,3,4,5), outputs_idx=(7,), tau=1, with_y=True):

    assert(K >= 1 and tau >= 1)

    D = df.to_numpy()
    N = D.shape[0] - K - tau

    assert(N >= K)

    lst_x = []
    lst_y = []

    for i in range(K, N - (tau - 1), skip_ts):

        xi = D[(i-K):(i + tau - 1):stride, inputs_idx].flatten()

        if with_y:
            xi = np.concatenate((xi, D[(i-K):i:stride, outputs_idx].flatten()))

        lst_x.append(torch.Tensor(xi))

        yi = D[(i+tau-1),outputs_idx]
        lst_y.append(torch.Tensor(yi))

    X = torch.stack(lst_x)
    Y = torch.stack(lst_y)

    return(X,Y)


def load_tensor_from_csv(path_name, K):
    L = load_dfs(path_name)

    lst_x = []
    lst_y = []

    for j in range(0,len(L)):
        (Xj,Yj) = dataset_df_to_tensor(L[j],K)
        lst_x.append(Xj)
        lst_y.append(Yj)

    X = torch.cat(lst_x, 0)
    Y = torch.cat(lst_y, 0)

    return (X,Y)

