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
from .model import load_model
from .utils import make_dir

    
def train(model, 
        train_loader, valid_loader, 
        criterion, optimizer, scheduler, 
        n_epoch, early_stop, patience, 
        path_string, file_string, use_gpu=False):

    history = History(model.target_type_string)

    early_stop_counter = 0
    last_valid_loss = math.inf
    best_model_save_string = ''
    lr_a = _get_LR(optimizer)
    lr_b = _get_LR(optimizer)

    for i in range(n_epoch):
        start = time.time() #start of epoch time
        do_epoch(model, train_loader, criterion, optimizer, use_gpu)
        end = time.time() #end of epoch time
        epoch_training_time = end - start

        train_loss, train_metrics = validate(model, train_loader, criterion, use_gpu)
        valid_loss, valid_metrics = validate(model, valid_loader, criterion, use_gpu)
        
        history.save_step(train_loss, valid_loss, train_metrics, valid_metrics, epoch_training_time)
        terminal_printer(model.target_type_string, i, epoch_training_time, train_loss, valid_loss, train_metrics, valid_metrics)

        lr_a = _get_LR(optimizer) #lr before scheduler step
        scheduler.step(valid_loss)
        lr_b = _get_LR(optimizer) #lr after scheduler step

        if lr_a != lr_b: #reload old best model if lr changed
            print('Changing to Last Best Model')
            load_model(model, best_model_save_string)

        if valid_loss < last_valid_loss: #if new best model
            last_valid_loss = valid_loss
            print('New Model Saved (New Best:{:.4f})\n'.format(valid_loss))
            early_stop_counter = 0
            torch_extension = '.torch'
            model_file_string = file_string + '_epoch{}_validloss{:.4f}_'.format(i, valid_loss)
            folder_path = make_dir(os.path.join('.', 'saved_models', path_string))
            best_model_save_string = os.path.join(folder_path, model_file_string + torch_extension)
            #print(best_model_save_string)
            torch.save(model.state_dict(), best_model_save_string)
        else: #if not a new bets model
            early_stop_counter += 1
            print("No improvement, in the last {} epoch(s), on the valid loss (Best:{:.4f}, Current:{:.4f})\n".format(
                early_stop_counter, last_valid_loss, valid_loss))
            if early_stop_counter >= early_stop:
                print("Early Stop dépassé! Valid loss de {:.4f} atteinte".format(last_valid_loss))
                break

    return history, best_model_save_string

def terminal_printer(target_type_string, epoch, epoch_training_time, train_loss, valid_loss, train_metrics, valid_metrics):
    if target_type_string == 'Classification':
        print('Epoch {} - Training time: {:.2f}s - Train loss: {:.4f} - Valid loss: {:.4f} - Train F1: {:.2f} - Valid F1: {:.2f} - Train acc: {:.2f} - Valid acc: {:.2f}'.format(
            epoch, epoch_training_time, train_loss, valid_loss, train_metrics.f1, valid_metrics.f1, train_metrics.accuracy, valid_metrics.accuracy))
    elif target_type_string == 'Regression':
        print('Epoch {} - Training time: {:.2f}s - Train loss: {:.4f} - Valid loss: {:.4f} - Train R2: {:.4f} - Valid R2: {:.4f} - Train MSE: {:.4f} - Valid MSE: {:.4f} - Train MAE: {:.4f} - Valid MAE: {:.4f}'.format(
            epoch, epoch_training_time, train_loss, valid_loss, train_metrics.r2_score, valid_metrics.r2_score, train_metrics.mean_square_error, valid_metrics.mean_square_error, train_metrics.mean_abs_error, valid_metrics.mean_abs_error))

def _get_LR(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
            lr = float(param_group['lr'])
    return lr

def do_epoch(model, train_loader, criterion, optimizer, use_gpu):
    model.train()
    for j, batch in enumerate(train_loader):
        inputs, targets = batch

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

def validate(model, valid_loader, criterion, use_gpu=False):
    true = []
    pred = []
    valid_loss = []

    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(valid_loader):
            inputs, targets = batch
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)

            predictions = output.max(dim=1)[1]

            valid_loss.append(criterion(output, targets).item())
            true.extend(targets.data.cpu().numpy().tolist())
            pred.extend(predictions.data.cpu().numpy().tolist())

    loss = sum(valid_loss) / len(valid_loss)
    metrics = Metrics(true, pred, model.target_type_string)

    return loss, metrics

def test(model, test_loader, criterion, use_gpu=False):
    test_loss, test_metrics = validate(model, test_loader, criterion, use_gpu)
    if model.target_type_string == 'Classification':
        print('Test:\n\tLoss: {}\n\tAccuracy: {}\n\tF1 Score (0.5): {}'.format(test_loss, test_metrics.accuracy, test_metrics.f1))
    elif model.target_type_string == 'Regression':
        print('Test:\n\tLoss: {}\n\tR2: {}\n\tMSE: {}\n\tMAE: {}'.format(test_loss, test_metrics.r2_score, test_metrics.mean_square_error, test_metrics.mean_abs_error))
    return test_loss, test_metrics
