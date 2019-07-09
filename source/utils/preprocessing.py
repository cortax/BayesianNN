import glob
import os
import math
import pandas as pd
import numpy as np


def sortKeyFunc(s):
    return int(os.path.basename(s)[12:-4])

def split_dataframe_on_ratio(dataframe, ratio):
    rows, cols = dataframe.shape
    index_where_to_split = math.floor(rows * ratio)
    dataframe_part_A = dataframe.iloc[:index_where_to_split]
    dataframe_part_B = dataframe.iloc[index_where_to_split:]
    
    return dataframe_part_A, dataframe_part_B

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