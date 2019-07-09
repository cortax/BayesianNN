import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from source.utils.preprocessing import *
from source.utils.utils import is_float
from source.utils.storage import DataStorage

class NoFeatureIndexOnlyDataset(data.Dataset):
    def __init__(self, dataset, signal_type):
        self.windowed_dataset = dataset
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
            self.signal_type_string = 'Directional'

        #print('len dataset', len(dataset))
        #print('dataset', dataset)

    def __getitem__(self, index):
        features_window, labels = self.windowed_dataset[index]

        #print('features_window:', features_window)
        #print('features_window, 1st timestamp, 1st feature:', features_window[0][0])
        #print('labels:', labels)

        #On garde juste le close pour le moment
        #returns_close = np.delete(features_window, 0, axis=1).reshape(1, -1)
        #returns_close = torch.Tensor(returns_close)
        #returns_close = torch.squeeze(torch.Tensor(returns_close))

        label = labels[self.signal_type - 1]
        #print('label:', label)
        if self.signal_type_string != 'Regression':
            label = int(label)
        elif self.signal_type_string == 'Regression':
            label = np.float32(label)
        return torch.Tensor(features_window), label

    def __len__(self):
        return len(self.windowed_dataset)

def _split_on_ratio(returns, signals, split_train_test_ratio=0.8):
        returns_train, returns_test = split_dataframe_on_ratio(returns, split_train_test_ratio)
        signals_train, signals_test = split_dataframe_on_ratio(signals, split_train_test_ratio)
        #print(returns_train.tail())
        #print(returns_test.head())
        return returns_train, returns_test, signals_train, signals_test

def _merge_features_windows_with_label(features_windows, labels_windows):
    #print('Inside Merger')
    #print(features_windows)
    merged = []
    for window in range(len(features_windows)):
        #print(window)
        datatuple = (features_windows[window], labels_windows[window][-1])
        #print('merge test:', labels_windows[window][-1])
        #print('testtuple:', datatuple)
        merged.append(datatuple)
    return merged

def _normalize_df(df, avg_open, std_open, avg_close, std_close):
    _df = df.transform(lambda row: row_normalize(row, avg_open, std_open, avg_close, std_close), axis=1)
    return _df

def row_normalize(row, avg_open, std_open, avg_close, std_close):
    #print(row)
    row['Open'] = (row['Open'] - avg_open)/std_open
    row['Close'] = (row['Close'] - avg_close)/std_close
    return row  

def get_single_feature_multiple_datasets(number_subdatasets, size_windows, signal_type,
                features_path='./datasets/S&P500/S&P500_data_features.pkl',
                labels_path='./datasets/S&P500/S&P500_data_labels.pkl',
                split_train_test_ratio=0.8, split_train_valid_ratio=0.8, 
                normalize=False,
                random_seed=42):

    print('Loading Data')
    loaded_features = pd.read_pickle(features_path)
    #print(loaded_features)
    #print(loaded_features.columns)
    loaded_labels = pd.read_pickle(labels_path)
    #print(loaded_labels)

    #keeping only "base data" and absolute diff and returns
    loaded_features=loaded_features[['Close_diff_pct_1']]

    #contiguous splits
    list_chunks_features = np.array_split(loaded_features, number_subdatasets)
    list_chunks_labels = np.array_split(loaded_labels, number_subdatasets)

    list_train_dataset = []
    list_valid_dataset = []
    list_test_dataset = []
    list_data_storage = []
    for chunk in range(number_subdatasets):
        print('For Dataset Chunk #:', chunk)
        data_storage = DataStorage() #Used for Strategies and Portfolios
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
            """
            avg_close, std_close = get_normalization_data(returns_train['Close'], degrees_of_liberty=1)
            avg_open, std_open = get_normalization_data(returns_train['Open'], degrees_of_liberty=1)

            returns_train = _normalize_df(returns_train, avg_open, std_open, avg_close, std_close)
            returns_valid = _normalize_df(returns_valid, avg_open, std_open, avg_close, std_close)
            returns_test = _normalize_df(returns_test, avg_open, std_open, avg_close, std_close)
            """

        print('Making Rolling Windows')
        features_windows_train = get_sliding_window(features_train, window_size=size_windows)
        features_windows_valid = get_sliding_window(features_valid, window_size=size_windows)
        features_windows_test = get_sliding_window(features_test, window_size=size_windows)
        #print(np.shape(features_windows_test))
        #print(features_windows_test[-1][-1][features_test.columns.get_loc('Close_diff_pct_1')])

        labels_windows_train = get_sliding_window(labels_train, window_size=size_windows)
        labels_windows_valid = get_sliding_window(labels_valid, window_size=size_windows)
        labels_windows_test = get_sliding_window(labels_test, window_size=size_windows)

        print('Merging Features and Signal')
        _train = _merge_features_windows_with_label(features_windows_train, labels_windows_train)
        _valid = _merge_features_windows_with_label(features_windows_valid, labels_windows_valid)
        _test = _merge_features_windows_with_label(features_windows_test, labels_windows_test)

        print('Making Pytorch Dataset Objects')
        train_dataset = NoFeatureIndexOnlyDataset(_train, signal_type)
        valid_dataset = NoFeatureIndexOnlyDataset(_valid, signal_type)
        test_dataset = NoFeatureIndexOnlyDataset(_test, signal_type)

        list_train_dataset.append(train_dataset)
        list_valid_dataset.append(valid_dataset)
        list_test_dataset.append(test_dataset)

        #saving some useful information (returns de close en test)
        test_returns_close_pct = []
        for w, windows in enumerate(features_windows_test):
            returns = windows[-1][features_test.columns.get_loc('Close_diff_pct_1')]
            test_returns_close_pct.append(round(returns,6))
        #simpler method would be to fetch the last len(pred or true) data from features_test
        data_storage.save_test_returns_pct(test_returns_close_pct)
        
        list_data_storage.append(data_storage)

    return list_train_dataset, list_valid_dataset, list_test_dataset, list_data_storage  

def single_feature_multiple_datasets_main(dataset_name_string,
                                    number_subdatasets,
                                    split_train_test_ratio=0.8,
                                    split_train_valid_ratio=0.8,
                                    size_windows=40,
                                    random_seed=42,
                                    labels_type=4,
                                    normalize=False
                                    ):
    print('dataset_name_string:', dataset_name_string)
    print('number_subdatasets:', number_subdatasets)
    
    if dataset_name_string == 'All':
        print('Not implemented yet')
        return None

    if dataset_name_string == 'S&P500':
        features_path = './datasets/S&P500/S&P500_data_features.pkl'
        labels_path='./datasets/S&P500/S&P500_data_labels.pkl'
        list_train_dataset, list_valid_dataset, list_test_dataset, list_data_storage = get_single_feature_multiple_datasets(
                number_subdatasets, size_windows, labels_type,
                features_path=features_path, labels_path=labels_path,
                split_train_test_ratio=split_train_test_ratio, split_train_valid_ratio=split_train_valid_ratio, 
                normalize=normalize,
                random_seed=random_seed)

        #premier itérateur: Une des tranches du dataset
        #deuxième itérateur: Les Tuple (série, label) dans le dataset
        #troisième itérateur: Les éléments du tuple (X ou Y)
        #quatrième itérateur (X seulement): les éléments de la série
        print('Testing Getter')
        print('Number of Datasets:', len(list_train_dataset))
        print('Size Dataset 0:', len(list_train_dataset[0]))
        print('Dataset 0 (Object):', list_train_dataset[0])
        print('Train Tuple 0:', list_train_dataset[0][0])
        #print('Shape X:', list_train_dataset[0][0][0].shape)
        print('Tuple X:', list_train_dataset[0][0][0])
        print('Tuple X, timestamp 0:', list_train_dataset[0][0][0][0])
        #print('Type X, timestamp 0:', type(list_train_dataset[0][0][0][0]))
        print('Tuple X, timestamp 0, feature 0:', list_train_dataset[0][0][0][0][0])
        print('Tuple Y:', list_train_dataset[0][0][1])
        #print('Type Y:', type(list_train_dataset[0][0][1]))