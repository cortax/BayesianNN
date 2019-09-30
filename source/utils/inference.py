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

from torchvision.transforms import ToTensor

from .history import History
from .metrics import Metrics
from .model import load_entire_model
from .utils import make_dir
from .training import validate
from .losses import loss_function_picker
from .preprocessing import *


def inference(csv_files_path, stride, 
            train_test_ratio, train_valid_ratio, 
            forecasting, feature_endo, feature_exo, 
            target_choice, shift_delta, 
            windows_size, 
            entire_model_string, target_type_string='Regression', 
            use_gpu=False):
    
    print('Preparing Data')

    list_df, rescaling_array = get_rescaled_shrinked_list_df(csv_files_path, stride)
    list_df_train, list_df_valid, list_df_test = get_train_valid_test_list_df(list_df, train_test_ratio, train_valid_ratio, shuffle=False, random_seed=42)

    #list_df_train_features, list_df_train_targets = get_features_and_targets(list_df_train, forecasting, feature_endo, feature_exo, target_choice, shift_delta)
    #list_df_valid_features, list_df_valid_targets = get_features_and_targets(list_df_valid, forecasting, feature_endo, feature_exo, target_choice, shift_delta)
    list_df_test_features, list_df_test_targets = get_features_and_targets(list_df_test, forecasting, feature_endo, feature_exo, target_choice, shift_delta)

    list_df_test_features_windows = list_df_to_overlapping_sliding_windows(list_df_test_features, windows_size, mode="append")
    list_df_test_targets_windows = list_df_to_overlapping_sliding_windows(list_df_test_targets, windows_size, mode="append")
    #print(list_df_test_features_windows)
    #print(list_df_test_targets_windows)

    print('Loading Model and Getting Criterion')
    model = load_entire_model(entire_model_string)
    #model.eval()
    loss_function = loss_function_picker(target_type_string)

    #iterate on all df, getting the predictions and the ground truths
    print('Running Inference')
    for d in range(len(list_df_test)):
        #TODO: Convert this list_df_test_features_windows and list_df_test_targets_windows into inference dataset object
        test_loader = tData.DataLoader(test_dataset)

        """
        with torch.no_grad():
            true = []
            pred = []
            print('d:', d)
            for w, window in enumerate(list_df_test_features_windows[d]):
                print('w, window', w, window)
                output = model(window.tolist())
                pred.extend(output)
                true.extend(list_df_test_targets_windows[d][w][-1].tolist())
            #print(true)
        """

        test_loss, test_metrics = validate(model, test_loader, loss_function, use_gpu)
        
        #TODO: Unprepare data in true and pred

        #TODO: Plot true vs pred
        
        """
        if model.target_type_string == 'Classification':
            print('Test:\n\tLoss: {}\n\tAccuracy: {}\n\tF1 Score (0.5): {}'.format(test_loss, test_metrics.accuracy, test_metrics.f1))
        elif model.target_type_string == 'Regression':
            print('Test:\n\tLoss: {}\n\tR2: {}\n\tMSE: {}\n\tMAE: {}'.format(test_loss, test_metrics.r2_score, test_metrics.mean_square_error, test_metrics.mean_abs_error))
        """
