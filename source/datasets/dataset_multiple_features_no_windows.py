import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

from source.utils.preprocessing import *
from source.utils.utils import is_float
from source.utils.storage import DataStorage

class FeaturesIndexOnlyNoWindowsDataset(data.Dataset):
    def __init__(self, dataset, labels, signal_type):
        self.no_window_dataset = dataset
        self.labels = labels
        self.signal_type = signal_type

        if signal_type == 1:
            self.signal_type_string = 'Regression'
        elif signal_type == 2:
            self.signal_type_string = 'Directional'
        elif signal_type == 3:
            self.signal_type_string = 'Trading'
        elif signal_type == 4:
            self.signal_type_string = 'Positional'
        else:
            self.signal_type = 2
            self.signal_type_string = 'Directional'

        #print('len dataset', len(dataset))
        #print('dataset', dataset)

    def __getitem__(self, index):
        features = self.no_window_dataset.iloc[index]
        #print('features:', features)
        label = self.labels.iloc[index][self.signal_type - 1]
        #print('label:', label)
        if self.signal_type_string != 'Regression':
            label = int(label) #cast from float to int if classification
        elif self.signal_type_string == 'Regression':
            label = np.float32(label)
        #print(torch.Tensor(features), label)
        return torch.Tensor(features), label

    def __len__(self):
        return len(self.no_window_dataset)

def _split_on_ratio(returns, signals, split_train_test_ratio=0.8):
        returns_train, returns_test = split_dataframe_on_ratio(returns, split_train_test_ratio)
        signals_train, signals_test = split_dataframe_on_ratio(signals, split_train_test_ratio)
        #print(returns_train.tail())
        #print(returns_test.head())
        return returns_train, returns_test, signals_train, signals_test
    
def _normalize_df(df, avg_open, std_open, avg_close, std_close):
    _df = df.transform(lambda row: row_normalize(row, avg_open, std_open, avg_close, std_close), axis=1)
    return _df

def row_normalize(row, avg_open, std_open, avg_close, std_close):
    #print(row)
    row['Open'] = (row['Open'] - avg_open)/std_open
    row['Close'] = (row['Close'] - avg_close)/std_close
    return row  

def get_multiple_features_no_window_multiple_datasets(number_subdatasets, signal_type,
                features_path='./datasets/S&P500/S&P500_data_features.pkl',
                labels_path='./datasets/S&P500/S&P500_data_labels.pkl',
                split_train_test_ratio=0.8, split_train_valid_ratio=0.8, 
                normalize=False,
                random_seed=42):

    print('Loading Data')
    loaded_features = pd.read_pickle(features_path)
    #print(loaded_features)
    loaded_labels = pd.read_pickle(labels_path)
    #print(loaded_labels)

    list_chunks_features = np.array_split(loaded_features, number_subdatasets)
    list_chunks_labels = np.array_split(loaded_labels, number_subdatasets)

    list_train_dataset = []
    list_valid_dataset = []
    list_test_dataset = []
    list_data_storage = []
    for chunk in range(number_subdatasets):
        print('For Dataset Chunk #:', chunk)
        data_storage = DataStorage()
        print('Splitting (Returns and Signals) on Ratios')
        #splitting chunk of data into train-test
        features_train, features_test, labels_train, labels_test = _split_on_ratio(list_chunks_features[chunk], list_chunks_labels[chunk], split_train_test_ratio)
        #print(features_test['Close_diff_pct_1'].head())
        data_storage.save_returns_pct_features_train_valid(features_train['Close_diff_pct_1'])
        data_storage.save_returns_pct_features_test(features_test['Close_diff_pct_1'])

        #splitting chunk of train data into train-valid
        features_train, features_valid, labels_train, labels_valid = _split_on_ratio(features_train, labels_train, split_train_valid_ratio)
        data_storage.save_returns_pct_features_train(features_train['Close_diff_pct_1'])
        data_storage.save_returns_pct_features_valid(features_valid['Close_diff_pct_1'])

        if normalize:
            print('Normalizing based on Training Data')
            #TODO: Repenser et refaire en fonction des 3000 colonnes

        print('Making Pytorch Dataset Objects')
        train_dataset = FeaturesIndexOnlyNoWindowsDataset(features_train, labels_train, signal_type)
        valid_dataset = FeaturesIndexOnlyNoWindowsDataset(features_valid, labels_valid, signal_type)
        test_dataset = FeaturesIndexOnlyNoWindowsDataset(features_test, labels_test, signal_type)

        list_train_dataset.append(train_dataset)
        list_valid_dataset.append(valid_dataset)
        list_test_dataset.append(test_dataset)

        #saving some useful information
        data_storage.save_test_returns_pct(features_test['Close_diff_pct_1'])
        
        list_data_storage.append(data_storage)

    return list_train_dataset, list_valid_dataset, list_test_dataset, list_data_storage
    