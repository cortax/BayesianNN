from pandas import read_csv
from os import listdir
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch

from torch.utils.data import DataLoader

def data_loader(X,y):
    return DataLoader(TensorDataset(X,y), batch_size=32)

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





