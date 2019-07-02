from pandas import read_csv
from os import listdir
import pandas as pd
import torch
import numpy as np


def load_dfs(path_name, columns=range(2,12)):

    assert(isinstance(path_name, str))

    if path_name[-1] == '/':
        path_name = path_name[:-1]

    file_names = listdir(path_name)

    dfs = []
    for file_name in file_names:
        if file_name.endswith('.csv'):
            dfs.append(read_csv(path_name + '/' + file_name, usecols=columns, skiprows=2))

    return dfs

def dataset_df_to_tensor(df, K):
    inputs_idx = [1,2,3,4,5]
    outputs_idx = [7]

    D = df.to_numpy()
    N = D.shape[0]-K

    lst_x = []
    lst_y = []
    for i in range(K,N):
        xi = D[(i-K):i,inputs_idx].flatten()
        lst_x.append(torch.Tensor(xi))
        yi = D[i,outputs_idx]
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





