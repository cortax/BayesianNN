import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from source.dataset.dataset_multiple_features_no_windows import get_multiple_features_no_window_multiple_datasets
from source.loaders.dataloaders import get_train_valid_test_generic_loaders, getWeightedRandomSampler
from source.utils.training import train, test
from source.utils.model import load_model
from source.models.fcl import IndexFCLModel
from source.utils.strategy import ModelStrategy, EchoStrategy, PureNaiveStrategy, BalancedDynamicNaiveStrategy, BalancedLookAheadNaiveStrategy
from source.utils.portfolio import Portfolio, save_pf_values
from source.utils.plot import histograms_lineGraphs_moments_pandas

def full_features_fcl_training(index_dataset, experiment_number, amount_datasets,
                    n_epoch, batch_size, sampler_type,
                    early_stop, patience, cooldown, factor, learning_rate, verbose, threshold,
                    features_amount, labels_type, normalize,
                    split_train_test_ratio, split_train_valid_ratio,
                    optim_type, weight_decay, momentum, dampening,
                    hidden_size_layer_1, hidden_size_layer_2, hidden_size_layer_3,
                    input_dropout, hidden_dropout,
                    random_seed, num_workers, use_gpu, gpu_device_id,
                    dataset_name_string
                    ):

    print('Start of Main')
    features_path = os.path.join('.', 'datasets', dataset_name_string, dataset_name_string +'_data_features.pkl')
    labels_path = os.path.join('.', 'datasets', dataset_name_string, dataset_name_string +'_data_labels.pkl')
    folder_string = 'featuresFCLModel/{}/experiment{}'.format(dataset_name_string, experiment_number)
    file_name_string = 'subDatasetNumber_{}of{}_TODO'.format(index_dataset+1, amount_datasets)
    
    print('Making Dataset')
    list_train_dataset, list_valid_dataset, list_test_dataset, list_data_storage = get_multiple_features_no_window_multiple_datasets(
            amount_datasets, labels_type,
            features_path ='./datasets/S&P500/S&P500_data_features.pkl',
            labels_path ='./datasets/S&P500/S&P500_data_labels.pkl',
            split_train_test_ratio=split_train_test_ratio, split_train_valid_ratio=split_train_valid_ratio, 
            normalize=normalize,
            random_seed=random_seed)
    train_dataset, valid_dataset, test_dataset, data_storage = list_train_dataset[index_dataset], list_valid_dataset[index_dataset], list_test_dataset[index_dataset], list_data_storage[index_dataset]

    print('Saving Histograms, Line Graphs and Central Moments')
    histograms_lineGraphs_moments_pandas(data_storage, folder_string, file_name_string)

    if sampler_type==None:
        print('Not using any Special Sampler')
        sampler = None
        shuffle_train = True
    elif sampler_type=='oversampling':
        print('Using Oversampling with Replacement')
        list_train_y = [elem[1] for elem in train_dataset]
        #print('list_train_y:', list_train_y)
        sampler = getWeightedRandomSampler(list_train_y)
        shuffle_train = False
    
    print('Making Loaders')
    train_loader, valid_loader, test_loader = get_train_valid_test_generic_loaders(train_dataset, valid_dataset, test_dataset,
                batch_size=batch_size, shuffle_train=shuffle_train, train_sampler=sampler,
                random_seed=random_seed, num_workers=num_workers, pin_memory=False)

    print('Making Model')
    model = IndexFCLModel(features_amount, labels_type, 
                        hidden_size_layer_1, hidden_size_layer_2, hidden_size_layer_3,
                        input_dropout, hidden_dropout, bias=True)

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
    
    print('Start of Training')
    history, best_model_save_string = train(model, 
                                    train_loader, valid_loader, 
                                    criterion, optimizer, scheduler,
                                    n_epoch, early_stop, patience,
                                    folder_string, file_name_string, use_gpu=use_gpu)
    
    print('Saving Training History')
    history.save_display(folder_string, file_name_string)

    print('Start of Testing')
    model = load_model(model, best_model_save_string)
    test_loss, test_metrics = test(model, test_loader, 
                                criterion, use_gpu=use_gpu)
                                
    if labels_type != 1:    
    #saving conf mat to disk
        list_labels = ['0', '1', '2', '3']
        confMat_title = 'Confusion Matrix for FCL Base Model on Test Set #{}'.format(index_dataset)
        test_metrics.save_conf_matrix(list_labels, folder_string, file_name_string, title=confMat_title, normalize_matrix=False)

        print('Creating Strategies')
        pred_y = test_metrics.pred
        true_y = test_metrics.true

        model_strat = ModelStrategy(pred_y, labels_type)
        echo_strat = EchoStrategy(true_y, labels_type)
        pure_naive_strat = PureNaiveStrategy(labels_type, len(true_y))
        balanced_dynamic_naive_strat = BalancedDynamicNaiveStrategy(labels_type, true_y)
        balanced_lookahead_naive_strat = BalancedLookAheadNaiveStrategy(labels_type, true_y)

        print('Creating Portfolio')
        pf_model = Portfolio(data_storage.test_returns_pct_windows, model_strat.actions)
        pf_echo = Portfolio(data_storage.test_returns_pct_windows, echo_strat.actions)
        pf_pure = Portfolio(data_storage.test_returns_pct_windows, pure_naive_strat.actions)
        pf_bal_dynamic = Portfolio(data_storage.test_returns_pct_windows, balanced_dynamic_naive_strat.actions)
        pf_bal_lookahead = Portfolio(data_storage.test_returns_pct_windows, balanced_lookahead_naive_strat.actions)

        print('Saving Portfolio Values to CSV')
        data_dict = {'model': pf_model.values, 'echo':pf_echo.values, 'pure': pf_pure.values, 'dynamic': pf_bal_dynamic.values, 'lookahead': pf_bal_lookahead.values}
        save_pf_values(folder_string, file_name_string, data_dict)

    print('End of Main')
    return test_loss, test_metrics

def train_features_fcl_main(experiment_number, amount_datasets,
                    n_epoch, batch_size, sampler_type,
                    early_stop, patience, cooldown, factor, learning_rate, verbose, threshold,
                    features_amount, labels_type, normalize,
                    split_train_test_ratio, split_train_valid_ratio,
                    optim_type, weight_decay, momentum, dampening,
                    hidden_size_layer_1, hidden_size_layer_2, hidden_size_layer_3,
                    input_dropout, hidden_dropout,
                    random_seed, num_workers, use_gpu, gpu_device_id,
                    dataset_name_string):
    
    list_test_loss = []
    list_test_metrics = []
    for index_dataset in range(amount_datasets):
        #print(index_dataset)
        test_loss, test_metrics = full_features_fcl_training(index_dataset, experiment_number, amount_datasets,
                    n_epoch, batch_size, sampler_type,
                    early_stop, patience, cooldown, factor, learning_rate, verbose, threshold,
                    features_amount, labels_type, normalize,
                    split_train_test_ratio, split_train_valid_ratio,
                    optim_type, weight_decay, momentum, dampening,
                    hidden_size_layer_1, hidden_size_layer_2, hidden_size_layer_3,
                    input_dropout, hidden_dropout,
                    random_seed, num_workers, use_gpu, gpu_device_id,
                    dataset_name_string)

        list_test_loss.append(test_loss)
        list_test_metrics.append(test_metrics)
    #TODO: Caluler la moyenne des résultats en test pour les sub datasets