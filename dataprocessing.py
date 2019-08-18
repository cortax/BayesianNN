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


class DatasetBasic(Dataset):

    def __init__(self, dfs, past_steps, future_steps, stride, skip_ts, smoothing):

        self.past_steps = past_steps
        self.future_steps = future_steps

        inputs_idx = (0,2,3,4,5,8)
        outputs_idx = (6)

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

