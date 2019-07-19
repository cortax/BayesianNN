import glob
import os
import math
import torch
import random
import pandas as pd
import numpy as np


#list of dataframes into non overlapping sliding windows
def list_df_to_contiguous_sliding_windows_arrays(list_df, window_size, rem_beg=True):
    all_windows = []
    for dfs in list_df:
        windows = df_to_contiguous_sliding_windows_arrays(dfs, window_size, rem_beg=True)
        all_windows.extend(windows)
    return all_windows

#single dataframe into non overlapping sliding windows
def df_to_contiguous_sliding_windows_arrays(df, window_size, rem_beg=True):
    amount_to_remove = len(df)%window_size
    if rem_beg:
        df_rem = remove_df_starting_rows(df, amount_to_remove)
    elif not rem_beg:
        df_rem = remove_df_ending_rows(df, amount_to_remove)
    
    rows = []
    windows = []
    for index, row in df_rem.iterrows():
        rows.append(row.values.tolist())
        if len(rows)==window_size:
            windows.append(rows)
            rows = []
    return windows

#list of dataframes into overlapping sliding windows
def list_df_to_overlapping_sliding_windows_arrays(list_df, window_size):
    all_windows = []
    for dfs in list_df:
        windows = df_to_overlapping_sliding_windows_arrays(dfs, window_size)
        all_windows.extend(windows)
    return all_windows

#single dataframe into overlapping sliding windows
def df_to_overlapping_sliding_windows_arrays(dataframe, window_size):
    values_array = dataframe.values                 
    s0, s1 = values_array.strides
    row, col = values_array.shape
    windows = np.lib.stride_tricks.as_strided(values_array, shape=(row-window_size+1, window_size, col), strides=(s0, s0, s1))
    return windows

#function to remove N starting rows from a dataframe
def remove_df_starting_rows(dataframe, amount_to_remove):
    if amount_to_remove > 0:
        df = dataframe.iloc[amount_to_remove:]                 
        return df
    else:
        return dataframe

#function to remove N trailing rows from a dataframe
def remove_df_ending_rows(dataframe, amount_to_remove):
    if amount_to_remove > 0:
        df = dataframe.iloc[:-amount_to_remove]                 
        return df
    else:
        return dataframe

#function for ordering files. Very basic (hardcoded positions)
def sortKeyFunc(s):
    return int(os.path.basename(s)[12:-4])

#dataframe rows into a vector
def df_rows_to_vectors(df):
    vectors = []
    for rows in range(len(df)):
        row = df.iloc[rows]
        vectors.append(row.values.tolist())
    return vectors

#splitting a pythong list or ndarray into 2 parts according to specificied ratio
def split_list_on_ratio(liste, ratio, shuffle=False, random_seed=42):
    len_list = len(liste)
    if shuffle:
        random.seed(random_seed)
        random.shuffle(liste)
    
    index_where_to_split = math.floor(len_list * ratio)

    list_part_a = liste[:index_where_to_split]
    list_part_b = liste[index_where_to_split:]
    
    return list_part_a, list_part_b

#loading multiple csv into multiple dataframes, where each csv is a dataframe. Partly hardcoded for a bombardier flight test CSV files.
def bomb_csv_to_df(csv_stringLoader):
    list_df = []
    list_data_units = []
    list_data_label_type = []
    allFiles = sorted(glob.glob(csv_stringLoader), key=sortKeyFunc)

    for files in allFiles:
        print('Loading:{}'.format(files))
        df = pd.read_csv(files)
        df = df.drop('Description', axis=1)
        df = df.set_index('TIME OF DAY IN SECONDS')

        data_units = df.iloc[0]
        data_units.name = 'Unit'

        data_label_type = df.iloc[1]
        data_label_type.name = 'Type'

        df = df.iloc[3:].reset_index()
        
        list_df.append(df)
        list_data_units.append(data_units)
        list_data_label_type.append(data_label_type)
        
    return list_df, list_data_units, list_data_label_type

#TODO: Func to verify if list_data_units and list_data_label_type are all the same

#purely exogeneous (non-regressive)
def list_df_to_exogeneous_df(list_df, target_choice):
    list_df_features = []
    list_df_targets = []

    for df in list_df:
        if target_choice==0:
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', '2nd Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow', '2nd Underfloor flow']
            targets_list = ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
        elif target_choice==1:
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        elif target_choice==2:
            features_list = ['TIME OF DAY IN SECONDS', '2nd Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '2nd Underfloor flow']
            targets_list = ['2nd AVIONICS BAY BULK TEMP']
        else:
            target_choice=1
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        
        df_features = df[features_list]
        df_targets = df[targets_list]
        
        list_df_features.append(df_features)
        list_df_targets.append(df_targets)
        
    return list_df_features, list_df_targets

#purely endogenous (non-autoregressive)
#to verify if the model can learn copying data
def list_df_to_endogeneous_df(list_df, target_choice):
    list_df_features = []
    list_df_targets = []

    for df in list_df:
        if target_choice==0:
            features_list = ['TIME OF DAY IN SECONDS', '1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
        elif target_choice==1:
            features_list = ['TIME OF DAY IN SECONDS', '1st AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        elif target_choice==2:
            features_list = ['TIME OF DAY IN SECONDS', '2nd AVIONICS BAY BULK TEMP']
            targets_list = ['2nd AVIONICS BAY BULK TEMP']
        else:
            target_choice=1
            features_list = ['TIME OF DAY IN SECONDS', '1st AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        
        df_features = df[features_list]
        df_targets = df[targets_list]
        
        list_df_features.append(df_features)
        list_df_targets.append(df_targets)
        
    return list_df_features, list_df_targets

#exogeneous & endogenous (non-autoregressive)
#to verify if the model can learn copying data with more data
def list_df_to_endo_exo_df(list_df, target_choice):
    list_df_features = []
    list_df_targets = []

    for df in list_df:
        if target_choice==0:
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', '2nd Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow', '2nd Underfloor flow'] + ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
        elif target_choice==1:
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow'] + ['1st AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        elif target_choice==2:
            features_list = ['TIME OF DAY IN SECONDS', '2nd Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '2nd Underfloor flow'] + ['2nd AVIONICS BAY BULK TEMP']
            targets_list = ['2nd AVIONICS BAY BULK TEMP']
        else:
            target_choice=1
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow'] + ['1st AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        
        df_features = df[features_list]
        df_targets = df[targets_list]
        
        list_df_features.append(df_features)
        list_df_targets.append(df_targets)
        
    return list_df_features, list_df_targets

#forecasting (endogeneous)
def list_df_forecasting_endo_df(list_df, target_choice, shift_delta=1):
    list_df_features = []
    list_df_targets = []

    for df in list_df:
        if target_choice==0:
            features_list = ['TIME OF DAY IN SECONDS', '1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
        elif target_choice==1:
            features_list = ['TIME OF DAY IN SECONDS', '1st AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        elif target_choice==2:
            features_list = ['TIME OF DAY IN SECONDS', '2nd AVIONICS BAY BULK TEMP']
            targets_list = ['2nd AVIONICS BAY BULK TEMP']
        else:
            target_choice=1
            features_list = ['TIME OF DAY IN SECONDS', '1st AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        
        df_features = df[features_list]
        df_targets = df[targets_list].shift(-shift_delta)
        
        df_features = df_features.drop(df_features.tail(shift_delta).index)
        df_targets = df_targets.drop(df_targets.tail(shift_delta).index)
        
        list_df_features.append(df_features)
        list_df_targets.append(df_targets)
        
    return list_df_features, list_df_targets

#forecasting (exogeneous)
def list_df_forecasting_exo_df(list_df, target_choice, shift_delta=1):
    list_df_features = []
    list_df_targets = []

    for df in list_df:
        if target_choice==0:
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', '2nd Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow', '2nd Underfloor flow']
            targets_list = ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
        elif target_choice==1:
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        elif target_choice==2:
            features_list = ['TIME OF DAY IN SECONDS', '2nd Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '2nd Underfloor flow']
            targets_list = ['2nd AVIONICS BAY BULK TEMP']
        else:
            target_choice=1
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        
        df_features = df[features_list]
        df_targets = df[targets_list].shift(-shift_delta)
        
        df_features = df_features.drop(df_features.tail(shift_delta).index)
        df_targets = df_targets.drop(df_targets.tail(shift_delta).index)
        
        list_df_features.append(df_features)
        list_df_targets.append(df_targets)
        
    return list_df_features, list_df_targets

#forecasting (endogeneous & exogeneous)
def list_df_forecasting_endo_exo_df(list_df, target_choice, shift_delta=1):
    list_df_features = []
    list_df_targets = []

    for df in list_df:
        if target_choice==0:
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', '2nd Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow', '2nd Underfloor flow'] + ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
        elif target_choice==1:
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow'] + ['1st AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        elif target_choice==2:
            features_list = ['TIME OF DAY IN SECONDS', '2nd Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '2nd Underfloor flow'] + ['2nd AVIONICS BAY BULK TEMP']
            targets_list = ['2nd AVIONICS BAY BULK TEMP']
        else:
            target_choice=1
            features_list = ['TIME OF DAY IN SECONDS', '1st Cooling Sys MASS FLOW', 'ACS_Zone_Actual_Temperature', 'Outside Air Temperature_OAT', 'Pressure Altitude', 'Mach', '1st Underfloor flow'] + ['1st AVIONICS BAY BULK TEMP']
            targets_list = ['1st AVIONICS BAY BULK TEMP']
        
        df_features = df[features_list]
        df_targets = df[targets_list].shift(-shift_delta)
        
        df_features = df_features.drop(df_features.tail(shift_delta).index)
        df_targets = df_targets.drop(df_targets.tail(shift_delta).index)
        
        list_df_features.append(df_features)
        list_df_targets.append(df_targets)
        
    return list_df_features, list_df_targets
