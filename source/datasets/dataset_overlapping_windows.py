import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from source.utils.preprocessing import *


def get_overlapping_windows_datasets(csv_files_path, forecasting, feature_endo, feature_exo, target_choice, windows_size, shift_delta=1, stride=1, train_test_ratio=0.8, train_valid_ratio=0.8, shuffle=False, random_seed=42):
    #loading CSV
    shrinked_list_df, rescaling_array = get_rescaled_shrinked_list_df(csv_files_path, stride)
    #Note: Train test and valid are made on flights, not the amount of vectors in total!
    list_df_train, list_df_valid, list_df_test = get_train_valid_test_list_df(shrinked_list_df, train_test_ratio, train_valid_ratio, shuffle, random_seed)
    
    ow_ds_train = OverlappingWindowsDataset(list_df_train, forecasting, feature_endo, feature_exo, target_choice, windows_size, shift_delta)
    ow_ds_valid = OverlappingWindowsDataset(list_df_valid, forecasting, feature_endo, feature_exo, target_choice, windows_size, shift_delta)
    ow_ds_test = OverlappingWindowsDataset(list_df_test, forecasting, feature_endo, feature_exo, target_choice, windows_size, shift_delta)

    return ow_ds_train, ow_ds_valid, ow_ds_test

class OverlappingWindowsDataset(data.Dataset):
    def __init__(self, list_df, forecasting, feature_endo, feature_exo, target_choice, windows_size, shift_delta=1, target_type_string='Regression'):
        #target_choice is a parameter to pick if we use bay 1 (1), bay 2 (2) or both bays (0) as targets
        #Forecasting decides if the targets are N steps ahead(if True), or if we predict the current time step (if False)
        #Endo is if we want to use the bay temperature in the features
        #Exo is if we want to use the other data (the data that arent bay temp) in the features
        #Target type string is either Regression or Classification. Required for other objects down the training pipeline.
        self.target_type_string = target_type_string
        #Shift Delta is the parameter for forecasting that decides the N step ahead for target prediction
        
        if forecasting: #forecasting task (N step ahead)
            if feature_endo and not feature_exo:
                list_df_features, list_df_targets = list_df_forecasting_endo_df(list_df, target_choice, shift_delta)
            elif feature_exo and not feature_endo:
                list_df_features, list_df_targets = list_df_forecasting_exo_df(list_df, target_choice, shift_delta)
            elif feature_endo and feature_exo:
                list_df_features, list_df_targets = list_df_forecasting_endo_exo_df(list_df, target_choice, shift_delta)
            else:
                list_df_features, list_df_targets = list_df_forecasting_endo_exo_df(list_df, target_choice, shift_delta)

        if not forecasting: #Intra step prediction
            if feature_endo and not feature_exo:
                list_df_features, list_df_targets = list_df_to_endogeneous_df(list_df, target_choice)
            elif feature_exo and not feature_endo:
                list_df_features, list_df_targets = list_df_to_exogeneous_df(list_df, target_choice)
            elif feature_endo and feature_exo:
                list_df_features, list_df_targets = list_df_to_endo_exo_df(list_df, target_choice)
            else:
                list_df_features, list_df_targets = list_df_to_endo_exo_df(list_df, target_choice)

        all_overlapping_features_windows = list_df_to_overlapping_sliding_windows(list_df_features, windows_size)
        all_overlapping_targets_windows = list_df_to_overlapping_sliding_windows(list_df_targets, windows_size)
        
        self.features = all_overlapping_features_windows
        self.targets = all_overlapping_targets_windows

    def __getitem__(self, index):
        features_item = self.features[index].tolist()
        targets_item = self.targets[index][-1].tolist()
        
        #TODO: Verify if returns typage is "ok"
        return torch.Tensor(features_item), torch.Tensor(targets_item)

    def __len__(self):
        if len(self.features)==len(self.targets):
            return len(self.targets)
        else:
            raise('The dataset does not have one target per feature and vice versa')
        
    def features_size(self, index=0):
        return len(self.features[index][-1].tolist())

    def labels_size(self, index=0):
        return len(self.targets[index][-1].tolist())

class OverlappingWindowsInferenceDataset(data.Dataset):
    def __init__(self, list_df_index, list_df, forecasting, feature_endo, feature_exo, target_choice, windows_size, shift_delta=1, target_type_string='Regression'):
        #target_choice is a parameter to pick if we use bay 1 (1), bay 2 (2) or both bays (0) as targets
        #Forecasting decides if the targets are N steps ahead(if True), or if we predict the current time step (if False)
        #Endo is if we want to use the bay temperature in the features
        #Exo is if we want to use the other data (the data that arent bay temp) in the features
        #Target type string is either Regression or Classification. Required for other objects down the training pipeline.
        self.target_type_string = target_type_string
        #Shift Delta is the parameter for forecasting that decides the N step ahead for target prediction
        
        list_df_features, list_df_targets = get_features_and_targets(list_df, forecasting, feature_endo, feature_exo, target_choice, shift_delta)

        all_overlapping_features_windows = list_df_to_overlapping_sliding_windows(list_df_features, windows_size, mode='append')
        all_overlapping_targets_windows = list_df_to_overlapping_sliding_windows(list_df_targets, windows_size, mode='append')
        
        self.list_df_index = list_df_index
        self.features = all_overlapping_features_windows[list_df_index]
        self.targets = all_overlapping_targets_windows[list_df_index]


    def __getitem__(self, index):
        features_item = self.features[index].tolist()
        targets_item = self.targets[index][-1].tolist()
        
        #TODO: Verify if returns typage is "ok"
        return torch.Tensor(features_item), torch.Tensor(targets_item)

    def __len__(self):
        if len(self.features)==len(self.targets):
            return len(self.targets)
        else:
            raise('The dataset does not have one target per feature and vice versa')
        
    def features_size(self, index=0):
        return len(self.features[index][-1].tolist())

    def labels_size(self, index=0):
        return len(self.targets[index][-1].tolist())