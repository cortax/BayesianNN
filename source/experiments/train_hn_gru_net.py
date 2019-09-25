import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from source.utils.training import *
from source.utils.model import *
from source.models.grun import *
from source.datasets.dataset_overlapping_windows import *
from source.dataloaders.loaders import *
from source.utils.history import *
from source.utils.metrics import *


def hn_gru_net_main(experiment_number, 
                    n_epoch, batch_size, sampler_type,
                    early_stop, patience, cooldown, factor, learning_rate, verbose, threshold,
                    forecasting, feature_endo, feature_exo, target_choice, 
                    windows_size, shift_delta, stride,
                    train_test_ratio, train_valid_ratio,
                    optim_type, weight_decay, momentum, dampening,
                    hidden_size, num_layers, dropout_hidden, dropout_Hn,
                    random_seed, num_workers, use_gpu, gpu_device_id,
                    restart=False, restart_file_string=None
                    ):

    print('Start of Main')
    csv_path = os.path.join('.', 'DataBombardier', '2sec', 'flight_test_*.csv')
    folder_string = 'Hn_Gru_Net/experiment{}'.format(experiment_number)
    file_name_string = ''
    
    print('Making Dataset')
    train_dataset, valid_dataset, test_dataset = get_overlapping_windows_datasets(csv_path, 
            forecasting, feature_endo, feature_exo, target_choice, 
            windows_size, shift_delta, stride,
            train_test_ratio, train_valid_ratio, 
            shuffle=False, random_seed=42)

    if sampler_type==None:
        print('Not using any Special Sampler')
        sampler = None
        shuffle_train = True
    else:
        print('Sampler Type unrecognized, using None instead')
        sampler = None
        shuffle_train = True

    print('Making Loaders')
    train_loader, valid_loader, test_loader = get_train_valid_test_generic_loaders(train_dataset, valid_dataset, test_dataset,
                batch_size=batch_size, shuffle_train=shuffle_train, train_sampler=sampler,
                random_seed=random_seed, num_workers=num_workers, pin_memory=False)

    print('Making Model')
    model = GatedRecurrentUnitHnNet(train_dataset.features_size(), train_dataset.labels_size(),
                    windows_size, hidden_size, num_layers, 
                    target_type_string='Regression', 
                    bias=True, batch_first=True, 
                    bidirectional=False, 
                    dropout_hidden=dropout_hidden, dropout_Hn=dropout_Hn)
    
    print('Managing Cuda devices')
    if not torch.cuda.is_available() and use_gpu:
        print("Not Running on CUDA Device")
        use_gpu = False

    if use_gpu:
        model.cuda()
        torch.cuda.set_device(gpu_device_id) # Fix bug where memory is allocated on GPU0 when ask to take GPU1.
        device = torch.device('cuda:%d' % gpu_device_id)
        print('Running on GPU %d' % gpu_device_id)

    print('Making Optimizer, Scheduler and Getting Criterion')
    criterion = model.loss_function
    unfrozen_params = [p for p in model.parameters() if p.requires_grad]
    
    if optim_type=='sgd':
        optimizer = optim.SGD(unfrozen_params,
                            lr=learning_rate, momentum=momentum, 
                            dampening=dampening, weight_decay=weight_decay, 
                            nesterov=True)
    else: #optim_type=='adam':
        optimizer = optim.Adam(unfrozen_params, lr=learning_rate, 
                            betas=(0.9, 0.999), eps=1e-08, 
                            weight_decay=weight_decay, amsgrad=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                mode='min', factor=factor, patience=patience, 
                                verbose=verbose, threshold=threshold, threshold_mode='rel', 
                                cooldown=cooldown, min_lr=0, eps=1e-08)

    best_model_save_string = None
    if restart and restart_file_string is not None:
        restart_model_weights_string = os.path.join('.', 'saved_models', folder_string, restart_file_string)
        if os.path.exists(restart_model_weights_string):
            print('Restarting from Specified Model Weights')
            model = load_model(model, restart_model_weights_string)
            best_model_save_string = restart_model_weights_string
        else:
            print('Cannot Restart from Specified Model Weights')
    
    print('Start of Training')
    history, best_model_save_string = train(model, 
                                    train_loader, valid_loader, 
                                    criterion, optimizer, scheduler,
                                    n_epoch, early_stop, patience,
                                    folder_string, file_name_string, use_gpu=use_gpu)
    
    print('Saving Training History')
    history.history_display(folder_string, file_name_string)

    print('Start of Testing')
    model = load_model(model, best_model_save_string)
    test_loss, test_metrics = test(model, test_loader, 
                                criterion, use_gpu=use_gpu)
    print("\ntrue\n")
    print(test_metrics.true)
    print("\npred\n")
    print(test_metrics.pred)

    print('End of Main')
    return test_loss, test_metrics, best_model_save_string
