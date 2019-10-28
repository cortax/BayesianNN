import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tData
import torch.nn.functional as func
import sklearn.metrics as met
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor

from .history import History
from .metrics import Metrics
from .model import load_entire_model
from .utils import make_dir
from .training import validate
from .losses import loss_function_picker
from .preprocessing import *
from source.datasets.dataset_overlapping_windows import *


def bomb_overlapping_inference(csv_files_path, stride, 
            train_test_ratio, train_valid_ratio, 
            forecasting, feature_endo, feature_exo, 
            target_choice, shift_delta, 
            windows_size, 
            entire_model_string, 
            folder_string, file_name_string, 
            target_type_string='Regression', 
            use_gpu=False):
    
    print('Preparing Data')
    list_df, (rescaling_array, rescaling_col_names) = get_rescaled_shrinked_list_df(csv_files_path, stride)
    list_df_train, list_df_valid, list_df_test = get_train_valid_test_list_df(list_df, train_test_ratio, train_valid_ratio, shuffle=False, random_seed=42)
    rescaling_values = get_targets_rescaling_values(rescaling_array, rescaling_col_names, target_choice)

    print('Loading Model and Getting Criterion')
    model = load_entire_model(entire_model_string)
    loss_function = loss_function_picker(target_type_string, loss_name='mse')

    #iterate on all df, getting the predictions and the ground truths
    print('Running Inference...')
    print('For Test')
    forloop_inference(list_df_test, forecasting, feature_endo, feature_exo, 
                    target_choice, windows_size, shift_delta, 
                    model, loss_function, use_gpu, rescaling_values, 
                    folder_string, file_name_string + 'test')

    print('For Valid')
    forloop_inference(list_df_valid, forecasting, feature_endo, feature_exo, 
                        target_choice, windows_size, shift_delta, 
                        model, loss_function, use_gpu, rescaling_values, 
                        folder_string, file_name_string + 'valid')
    print('For Train')
    forloop_inference(list_df_train, forecasting, feature_endo, feature_exo, 
                    target_choice, windows_size, shift_delta, 
                    model, loss_function, use_gpu, rescaling_values, 
                    folder_string, file_name_string + 'train')

def forloop_inference(list_df, 
                    forecasting, feature_endo, feature_exo, 
                    target_choice, windows_size, shift_delta, 
                    model, loss_function, use_gpu, 
                    rescaling_values, 
                    folder_string, file_name_string):
    for idx in range(len(list_df)):
        overlapping_windows_inference_dataset = OverlappingWindowsInferenceDataset(idx, list_df, forecasting, feature_endo, feature_exo, target_choice, windows_size, shift_delta)
        loader = tData.DataLoader(overlapping_windows_inference_dataset)
        _, metrics = validate(model, loader, loss_function, use_gpu)
        
        #Unprepare data in true and pred
        true = [t*(rescaling_values+1) for t in metrics.true]
        pred = [p*(rescaling_values+1) for p in metrics.pred]
        rmse = get_rmse(true, pred)

        #Plot true vs pred
        title_string = f'_Index_{idx}_RMSE_{rmse:.2f}'
        inference_plot(true, pred, title_string, folder_string, file_name_string, options='save')

def pick_targets(target_choice):
    if target_choice==0:
        targets_names_list = ['1st AVIONICS BAY BULK TEMP', '2nd AVIONICS BAY BULK TEMP']
    elif target_choice==1:
        targets_names_list = ['1st AVIONICS BAY BULK TEMP']
    elif target_choice==2:
        targets_names_list = ['2nd AVIONICS BAY BULK TEMP']
    else:
        #target_choice=1
        targets_names_list = ['1st AVIONICS BAY BULK TEMP']
    return targets_names_list

def get_targets_rescaling_values(rescaling_array, rescaling_col_names, target_choice):
    targets_names_list = pick_targets(target_choice)
    index_of_names = []
    for col_name_idx, col_name in enumerate(targets_names_list):
        idx = rescaling_col_names.index(col_name)
        index_of_names.append(idx)
    rescaling_values = rescaling_array[index_of_names]
    return rescaling_values

def inference_plot(true, pred, title_string, path_string, file_string, options='save'):
    plt.clf()
    plt.ylabel('Temperature')
    plt.xlabel('Timesteps')
    plt.plot(true, label="True")
    plt.plot(pred, label="Pred")
    plt.title(title_string)
    plt.legend()
    #plt.show()
    if options == 'save':
        img_extension = '.png'
        base_name = '_inference' + title_string
        folder_path = make_dir(os.path.join('.', 'plots', path_string))
        save_string = os.path.join(folder_path, file_string + base_name + img_extension)
        plt.savefig(save_string)

def get_rmse(true, pred):
    list_mse = []
    for n in range(len(true)):
        diff_sq = (true[n] - pred[n])**2
        list_mse.append(diff_sq)
    rmse = np.sqrt(np.sum(list_mse))
    return rmse