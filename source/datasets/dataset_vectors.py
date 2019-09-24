import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from source.utils.preprocessing import *


def get_vectors_datasets(csv_files_path,
                            forecasting, feature_endo, feature_exo, target_choice, 
                            shift_delta, 
                            train_test_ratio, train_valid_ratio, 
                            shuffle=False, random_seed=42):
    #loading CSV
    list_df, list_data_units, list_data_label_type = bomb_csv_to_df(csv_files_path)
    rescaled_list_df = rescale_list_of_df(list_df)
    
    #Note: Train test and valid are made on flights, not the amount of vectors in total!
    
    #Splitting list of df into train-test
    list_df_train, list_df_test = split_list_on_ratio(rescaled_list_df, train_test_ratio, shuffle, random_seed)
    #Splitting list of train into train-valid
    list_df_train, list_df_valid = split_list_on_ratio(list_df_train, train_valid_ratio, shuffle, random_seed)
    
    sfv_ds_train = VectorsDataset(list_df_train, forecasting, feature_endo, feature_exo, target_choice, shift_delta)
    sfv_ds_valid = VectorsDataset(list_df_valid, forecasting, feature_endo, feature_exo, target_choice, shift_delta)
    sfv_ds_test = VectorsDataset(list_df_test, forecasting, feature_endo, feature_exo, target_choice, shift_delta)
    
    return sfv_ds_train, sfv_ds_valid, sfv_ds_test
    
class VectorsDataset(data.Dataset):
    def __init__(self, list_df, forecasting, feature_endo, feature_exo, target_choice, shift_delta=1, target_type_string='Regression'):
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

        all_vectors_features = list_df_rows_to_vectors(list_df_features)
        all_vectors_targets = list_df_rows_to_vectors(list_df_targets)
        
        self.features = all_vectors_features
        self.targets = all_vectors_targets

    def __getitem__(self, index):
        features_item = self.features[index]
        targets_item = self.targets[index]
        
        #TODO: Verify if returns typage is "ok"
        return torch.Tensor(features_item), torch.Tensor(targets_item)

    def __len__(self):
        if len(self.features)==len(self.targets):
            return len(self.targets)
        else:
            raise('The dataset does not have one target per feature and vice versa')

    def features_size(self, index=0):
        return len(self.features[index])

    def labels_size(self, index=0):
        return len(self.targets[index])