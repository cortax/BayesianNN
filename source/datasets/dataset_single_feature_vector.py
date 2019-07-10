import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from source.utils.preprocessing import *

class SingleFeatureVectorDataset(data.Dataset):
    def __init__(self, target_type_string, target_choice):
        #target_choice is a parameter that yield what forecasting problem we are on.
            #1 is for bay 1, 2 is for bay 2, 0 is for both bays.

        




        self.target_type_string = target_type_string

    def __getitem__(self, index):
        features_window, labels = self.windowed_dataset[index]

        if self.target_type_string == 'Classification':
            label = int(label)
        elif self.target_type_string == 'Regression':
            label = np.float32(label)
        return torch.Tensor(features_window), label

    def __len__(self):
        return len(self.windowed_dataset)