import torch
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os import sys
from json import dumps

from source.preprocessing.preprocess import simple_preprocess, features_preprocess

from source.dataset.dataset_single_feature_contiguous import single_feature_multiple_datasets_main, get_single_feature_multiple_datasets
from source.dataset.dataset_multiple_features_contiguous import multiple_features_multiple_datasets_main, get_multiple_features_multiple_datasets

from source.loaders.dataloaders import dataloaders_multiple_features_main, get_train_valid_test_generic_loaders

from source.models.gru import IndexGRUModel

from source.train.train_features_gru import train_features_gru_main
from source.train.train_base_gru import train_base_gru_main
from source.train.train_features_fcl import train_features_fcl_main
from source.train.train_base_lstm import train_base_lstm_main
from source.train.train_features_lstm import train_features_lstm_main

from source.utils.utils import is_float, is_int
from source.utils.training import test
from source.utils.model import load_model
from source.utils.plot import plot_confusion_matrix
from source.utils.strategy import strategy_main, strategy_test
from source.utils.portfolio import portfolio_main, portfolio_test


parser = argparse.ArgumentParser()
#required
parser.add_argument('-exp', '--EXPERIMENT_NUMBER', action='store', nargs='?',
                    default=0, required=True, type=int)
parser.add_argument('-ad', '--AMOUNT_DATASETS', action='store', nargs='?',
                    default=1, required=True, type=int)
parser.add_argument('-m', '--MODULE', action='store', nargs='?',
                    default='debug', required=True, type=str)

#non-required
parser.add_argument('-ml', '--MODEL', action='store', nargs='?',
                    default='features_gru', required=False, type=str)
parser.add_argument('-mm', '--MODULE_MODE', action='store', nargs='?',
                    default='', required=False, type=str)
parser.add_argument('-fa', '--FEATURES_AMOUNT', action='store', nargs='?', 
                    default=31, required=False, type=int) #features=31, base=1

#params for dataset and labels
parser.add_argument('--DATASET_NAME_STRING', action='store', nargs='?',
                    default='S&P500', required=False, type=str)
parser.add_argument('-lt', '--LABEL_TYPE', action='store', nargs='?', 
                    default=2, required=False, type=int)

parser.add_argument('-sw', '--SIZE_WINDOWS', action='store', nargs='?', 
                    default=40, required=False, type=int)
parser.add_argument('-bz', '--BATCH_SIZE', action='store', nargs='?', 
                    default=32, required=False, type=int)
parser.add_argument('-hs', '--HIDDEN_SIZE', action='store', nargs='?',
                    default=128, required=False, type=int)
parser.add_argument('-nl', '--NUM_LAYERS', action='store', nargs='?',
                    default=1, required=False, type=int)
parser.add_argument('-st', '--SAMPLER_TYPE', action='store', nargs='?',
                    default=None, required=False, type=str)

parser.add_argument('-stt', '--SPLIT_TRAIN_TEST_RATIO', action='store', nargs='?', 
                    default=0.8, required=False, type=float)
parser.add_argument('-stv', '--SPLIT_TRAIN_VALID_RATIO', action='store', nargs='?', 
                    default=0.8, required=False, type=float)
parser.add_argument('-n', '--NORMALIZE', action='store', nargs='?', 
                    default=False, required=False, type=bool)

#params for RNN dropout
parser.add_argument('--DROPOUT_HIDDEN', action='store', nargs='?', 
                    default=0.0, required=False, type=float)
parser.add_argument('--DROPOUT_LAYER', action='store', nargs='?', 
                    default=0.0, required=False, type=float)

#params for FCL
parser.add_argument('--HIDDEN_SIZE_LAYER_1', action='store', nargs='?',
                    default=31, required=False, type=int)
parser.add_argument('--HIDDEN_SIZE_LAYER_2', action='store', nargs='?',
                    default=10, required=False, type=int)
parser.add_argument('--HIDDEN_SIZE_LAYER_3', action='store', nargs='?',
                    default=5, required=False, type=int)
parser.add_argument('--INPUT_DROPOUT', action='store', nargs='?', 
                    default=0.1, required=False, type=float)
parser.add_argument('--HIDDEN_DROPOUT', action='store', nargs='?', 
                    default=0.5, required=False, type=float)

#params for seed and other hardware constraints
parser.add_argument('--RANDOM_SEED', action='store', nargs='?',
                    default=42, required=False, type=int)
parser.add_argument('-nw', '--NUM_WORKERS', action='store', nargs='?',
                    default=4, required=False, type=int)
parser.add_argument('--USE_GPU', action='store', nargs='?',
                    default=True, required=False, type=bool)
parser.add_argument('--GPU_DEVICE_ID', action='store', nargs='?',
                    default=0, required=False, type=int)

#params for some main functions
parser.add_argument('-mrpath', '--MODEL_RELATIVE_PATH', action='store', nargs='?',
                    default='', required=False, type=str)
parser.add_argument('--INDEX_DATASET', action='store', nargs='?',
                    default=0, required=False, type=int)

#params for optim and scheduler
parser.add_argument('-e', '--EPOCHS', action='store', nargs='?',
                    default=500, required=False, type=int)
parser.add_argument('-optim', '--OPTIM_TYPE', action='store', nargs='?', 
                    default='adam', required=False, type=str)
parser.add_argument('--EARLY_STOP', action='store', nargs='?', 
                    default=50, required=False, type=int)
parser.add_argument('--PATIENCE', action='store', nargs='?', 
                    default=10, required=False, type=int)
parser.add_argument('--COOLDOWN', action='store', nargs='?',
                    default=0, required=False, type=int)
parser.add_argument('--FACTOR', action='store', nargs='?', 
                    default=1/2, required=False, type=float)
parser.add_argument('-lr', '--LEARNING_RATE', action='store', nargs='?',
                    default=0.01, required=False, type=float)                    
parser.add_argument('--VERBOSE', action='store', nargs='?',
                    default=True, required=False, type=bool)
parser.add_argument('--THRESHOLD', action='store', nargs='?',
                    default=1e-6, required=False, type=float)
parser.add_argument('--WEIGHT_DECAY', action='store', nargs='?', 
                    default=0.0, required=False, type=float)
parser.add_argument('--MOMENTUM', action='store', nargs='?', 
                    default=0.9, required=False, type=float)
parser.add_argument('--DAMPENING', action='store', nargs='?', 
                    default=0.0, required=False, type=float)


def debug():
    labels_type = 3
    ar = [0, 1, 2, 3, 0, 0, 1, 2, 0, 1, 2, 2, 1, 2, 3, 2, 1, 0, 0]
    unique = np.unique(ar, return_index=False, return_inverse=False, return_counts=True, axis=None)
    print(unique[1])
    total = np.sum(unique[1])
    print(total)

    weight_vector = np.true_divide(unique[1], total)
    print(weight_vector)
    multinomial_result = np.random.multinomial(1, weight_vector)
    print(multinomial_result)
    random_label = (np.where(multinomial_result==1))[0][0]
    print(random_label)

def recall_test(index_dataset, experiment_number, amount_datasets, model_relative_path):
    #general
    n_epoch = 1000
    batch_size = 64

    #scheduling
    early_stop = 50
    patience = 10
    cooldown = 0
    factor = 1/2
    learning_rate = 0.010
    verbose = True
    threshold = 1e-5

    #data
    features_amount = 1800 #amount of columns in the dataframe
    labels_type = 2 #also called "amount_classes" #1 for regression, 2 for 2-classif(directionnal), 3 for 3-classif(trading), 4 for 4-classif(positional)
    size_windows = 40
    split_train_test_ratio = 0.8
    split_train_valid_ratio = 0.8
    normalize = False

    #optimizer
    optim_type = 'adam'
    #many (sgd + adam + other)
    weight_decay = 0
    #sgd
    momentum = 0.9
    dampening = 0
    #adam
    betas = (0.9, 0.999)
    eps = 1e-08

    #model
    hidden_size = 256
    num_layers = 1
    dropout_hidden = 0
    dropout_layer = 0

    #other
    random_seed = 42
    num_workers = 6

    #special strings
    dataset_name_string = 'S&P500'
    features_path ='./datasets/S&P500/S&P500_data_features.pkl'
    labels_path ='./datasets/S&P500/S&P500_data_labels.pkl'
    path_string = 'baseGruModel/{}/experiment{}'.format(dataset_name_string, experiment_number)
    file_string = 'subDatasetNumber|total:{}|{}_batchSize{}_learningRate{}_sizeWindow{}'.format(
                    index_dataset+1, amount_datasets, batch_size, learning_rate, size_windows)
    
    print('Making Dataset')
    list_train_dataset, list_valid_dataset, list_test_dataset = get_multiple_features_multiple_datasets(
            amount_datasets, size_windows, labels_type,
            features_path ='./datasets/S&P500/S&P500_data_features.pkl',
            labels_path ='./datasets/S&P500/S&P500_data_labels.pkl',
            split_train_test_ratio=split_train_test_ratio, split_train_valid_ratio=split_train_valid_ratio, 
            normalize=normalize,
            random_seed=random_seed)
    train_dataset, valid_dataset, test_dataset = list_train_dataset[index_dataset], list_valid_dataset[index_dataset], list_test_dataset[index_dataset]

    print('Making Loaders')
    train_loader, valid_loader, test_loader = get_train_valid_test_generic_loaders(train_dataset, valid_dataset, test_dataset,
                batch_size=batch_size,
                random_seed=random_seed, shuffle_train=True,
                num_workers=num_workers, pin_memory=False)

    print('Making Model')
    model = IndexGRUModel(features_amount, labels_type, size_windows, hidden_size, num_layers, 
                    bias=True, batch_first=True, 
                    dropout_hidden=dropout_hidden, bidirectional=False, 
                    dropout_layer=dropout_layer)

    print('Managing Cuda devices')
    use_gpu = torch.cuda.is_available()
    device_id = 0
    if use_gpu:
        model.cuda()
        torch.cuda.set_device(device_id) # Fix bug where memory is allocated on GPU0 when ask to take GPU1.
        device = torch.device('cuda:%d' % device_id)
        print('Running on GPU %d' % device_id)

    print('Start of Testing')
    model = load_model(model, model_relative_path)
    criterion = model.loss_function
    test_loss, test_metrics = test(model, test_loader, 
                                criterion, use_gpu=use_gpu)
    
    fig = plot_confusion_matrix(test_metrics.confusion_matrix, ['0', '1', '2', '3'], normalize=False, 
                            title='Confusion Matrix for GRU Base Model on Test Set #{}'.format(index_dataset))
    plt.show()

if __name__ == '__main__':
    #if no comment on what uses it, it is for "train"
    #TODO: The comments on what arg is used by what module
    args = parser.parse_args()

    MODULE = args.MODULE #all
    MODULE_MODE = args.MODULE_MODE #all
    MODEL = args.MODEL

    EXPERIMENT_NUMBER = args.EXPERIMENT_NUMBER #recall, train
    AMOUNT_DATASETS = args.AMOUNT_DATASETS #recall, train
    DATASET_NAME_STRING = args.DATASET_NAME_STRING #dataset

    INDEX_DATASET = args.INDEX_DATASET #recall
    MODEL_RELATIVE_PATH = args.MODEL_RELATIVE_PATH #recall

    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    SAMPLER_TYPE = args.SAMPLER_TYPE

    EARLY_STOP = args.EARLY_STOP
    PATIENCE = args.PATIENCE
    COOLDOWN = args.COOLDOWN
    FACTOR = args.FACTOR
    LEARNING_RATE = args.LEARNING_RATE
    VERBOSE = args.VERBOSE
    THRESHOLD = args.THRESHOLD

    FEATURES_AMOUNT = args.FEATURES_AMOUNT
    LABEL_TYPE = args.LABEL_TYPE
    SIZE_WINDOWS = args.SIZE_WINDOWS
    SPLIT_TRAIN_TEST_RATIO = args.SPLIT_TRAIN_TEST_RATIO
    SPLIT_TRAIN_VALID_RATIO = args.SPLIT_TRAIN_VALID_RATIO
    NORMALIZE = args.NORMALIZE

    HIDDEN_SIZE_LAYER_1 = args.HIDDEN_SIZE_LAYER_1
    HIDDEN_SIZE_LAYER_2 = args.HIDDEN_SIZE_LAYER_2
    HIDDEN_SIZE_LAYER_3 = args.HIDDEN_SIZE_LAYER_3
    INPUT_DROPOUT = args.INPUT_DROPOUT
    HIDDEN_DROPOUT = args.HIDDEN_DROPOUT

    OPTIM_TYPE = args.OPTIM_TYPE
    WEIGHT_DECAY = args.WEIGHT_DECAY
    MOMENTUM = args.MOMENTUM
    DAMPENING = args.DAMPENING

    HIDDEN_SIZE = args.HIDDEN_SIZE
    NUM_LAYERS = args.NUM_LAYERS
    DROPOUT_HIDDEN = args.DROPOUT_HIDDEN
    DROPOUT_LAYER = args.DROPOUT_LAYER

    RANDOM_SEED = args.RANDOM_SEED
    NUM_WORKERS = args.NUM_WORKERS
    USE_GPU = args.USE_GPU
    GPU_DEVICE_ID = args.GPU_DEVICE_ID

    print('\n----------ARGUMENTS----------\n')
    print(args, '\n')
    print('\n----------MODULE START----------\n')

    if MODULE=='debug':
        print('debug')
        debug()
    elif MODULE=='recall':
        print('Recalling is deprecated. It is barely using arg parser. Use with care')
        #TODO: Remake recall entirely
        #TODO: Make sure recall can restart mid training,.
        #TODO: Make sure recall saves differently
        #TODO: Make sure recall can reload hyper parameters
        recall_test(INDEX_DATASET, EXPERIMENT_NUMBER, AMOUNT_DATASETS, MODEL_RELATIVE_PATH)
    elif MODULE=='train' or MODULE=='model':
        print('train')
        if MODULE_MODE=='gru':
            if MODEL=='features_gru':
                train_features_gru_main(
                            EXPERIMENT_NUMBER, AMOUNT_DATASETS,
                            EPOCHS, BATCH_SIZE, SAMPLER_TYPE,
                            EARLY_STOP, PATIENCE, COOLDOWN, FACTOR, LEARNING_RATE, VERBOSE, THRESHOLD,
                            FEATURES_AMOUNT, LABEL_TYPE, SIZE_WINDOWS, NORMALIZE,
                            SPLIT_TRAIN_TEST_RATIO, SPLIT_TRAIN_VALID_RATIO,
                            OPTIM_TYPE, WEIGHT_DECAY, MOMENTUM, DAMPENING,
                            HIDDEN_SIZE, NUM_LAYERS, DROPOUT_HIDDEN, DROPOUT_LAYER,
                            RANDOM_SEED, NUM_WORKERS, USE_GPU, GPU_DEVICE_ID,
                            DATASET_NAME_STRING)
            elif MODEL=='base_gru':
                train_base_gru_main(EXPERIMENT_NUMBER, AMOUNT_DATASETS,
                            EPOCHS, BATCH_SIZE, SAMPLER_TYPE,
                            EARLY_STOP, PATIENCE, COOLDOWN, FACTOR, LEARNING_RATE, VERBOSE, THRESHOLD,
                            FEATURES_AMOUNT, LABEL_TYPE, SIZE_WINDOWS, NORMALIZE,
                            SPLIT_TRAIN_TEST_RATIO, SPLIT_TRAIN_VALID_RATIO,
                            OPTIM_TYPE, WEIGHT_DECAY, MOMENTUM, DAMPENING,
                            HIDDEN_SIZE, NUM_LAYERS, DROPOUT_HIDDEN, DROPOUT_LAYER,
                            RANDOM_SEED, NUM_WORKERS, USE_GPU, GPU_DEVICE_ID,
                            DATASET_NAME_STRING)
        elif MODULE_MODE=='lstm':
            if MODEL=='features_lstm':
                train_features_lstm_main(
                            EXPERIMENT_NUMBER, AMOUNT_DATASETS,
                            EPOCHS, BATCH_SIZE, SAMPLER_TYPE,
                            EARLY_STOP, PATIENCE, COOLDOWN, FACTOR, LEARNING_RATE, VERBOSE, THRESHOLD,
                            FEATURES_AMOUNT, LABEL_TYPE, SIZE_WINDOWS, NORMALIZE,
                            SPLIT_TRAIN_TEST_RATIO, SPLIT_TRAIN_VALID_RATIO,
                            OPTIM_TYPE, WEIGHT_DECAY, MOMENTUM, DAMPENING,
                            HIDDEN_SIZE, NUM_LAYERS, DROPOUT_HIDDEN, DROPOUT_LAYER,
                            RANDOM_SEED, NUM_WORKERS, USE_GPU, GPU_DEVICE_ID,
                            DATASET_NAME_STRING)
            elif MODEL=='base_lstm':
                train_base_lstm_main(EXPERIMENT_NUMBER, AMOUNT_DATASETS,
                            EPOCHS, BATCH_SIZE, SAMPLER_TYPE,
                            EARLY_STOP, PATIENCE, COOLDOWN, FACTOR, LEARNING_RATE, VERBOSE, THRESHOLD,
                            FEATURES_AMOUNT, LABEL_TYPE, SIZE_WINDOWS, NORMALIZE,
                            SPLIT_TRAIN_TEST_RATIO, SPLIT_TRAIN_VALID_RATIO,
                            OPTIM_TYPE, WEIGHT_DECAY, MOMENTUM, DAMPENING,
                            HIDDEN_SIZE, NUM_LAYERS, DROPOUT_HIDDEN, DROPOUT_LAYER,
                            RANDOM_SEED, NUM_WORKERS, USE_GPU, GPU_DEVICE_ID,
                            DATASET_NAME_STRING)        
        elif MODULE_MODE=='fcl':
            if MODEL=='features_fcl':
                train_features_fcl_main(EXPERIMENT_NUMBER, AMOUNT_DATASETS,
                            EPOCHS, BATCH_SIZE, SAMPLER_TYPE,
                            EARLY_STOP, PATIENCE, COOLDOWN, FACTOR, LEARNING_RATE, VERBOSE, THRESHOLD,
                            FEATURES_AMOUNT, LABEL_TYPE, NORMALIZE,
                            SPLIT_TRAIN_TEST_RATIO, SPLIT_TRAIN_VALID_RATIO,
                            OPTIM_TYPE, WEIGHT_DECAY, MOMENTUM, DAMPENING,
                            HIDDEN_SIZE_LAYER_1, HIDDEN_SIZE_LAYER_2, HIDDEN_SIZE_LAYER_3,
                            INPUT_DROPOUT, HIDDEN_DROPOUT,
                            RANDOM_SEED, NUM_WORKERS, USE_GPU, GPU_DEVICE_ID,
                            DATASET_NAME_STRING)
    elif MODULE=='strategy':
        print('stategy')
        if MODULE_MODE=='main':
            strategy_main()
        elif MODULE_MODE=='test':
            strategy_test()
    elif MODULE=='portfolio':
        print('portfolio')
        if MODULE_MODE=='main':
            portfolio_main()
        elif MODULE_MODE=='test':
            portfolio_test()
    elif MODULE=='preprocess':
        print('preprocess')
        if MODULE_MODE=='simple':
            print('Simple preprocess is deprecated. Use with care')
            simple_preprocess()
        elif MODULE_MODE=='features':
            print('features')
            features_preprocess()
    elif MODULE=='datasets' or MODULE=='dataset':
        print('dataset')
        if MODULE_MODE=='simple':
            if DATASET_NAME_STRING=='S&P500':
                single_feature_multiple_datasets_main(DATASET_NAME_STRING, 
                                        AMOUNT_DATASETS,
                                        SPLIT_TRAIN_TEST_RATIO,
                                        SPLIT_TRAIN_VALID_RATIO,
                                        SIZE_WINDOWS,
                                        RANDOM_SEED,
                                        NORMALIZE)
        elif MODULE_MODE=='features':
            if DATASET_NAME_STRING=='S&P500':
                multiple_features_multiple_datasets_main(DATASET_NAME_STRING,
                                                AMOUNT_DATASETS,
                                                SPLIT_TRAIN_TEST_RATIO,
                                                SPLIT_TRAIN_VALID_RATIO,
                                                SIZE_WINDOWS,
                                                RANDOM_SEED,
                                                LABEL_TYPE,
                                                NORMALIZE)
    elif MODULE=='dataloader' or MODULE=='dataloaders':
        print('dataloader')
        dataloaders_multiple_features_main()
    
    """
    with open("experiments.json", 'a') as experiments:
        info['args'] = {
            'USE_GPU': USE_GPU,
            'EPOCHS': EPOCHS,
            'PATIENCE': PATIENCE,
            'RANDOM_SEED': RANDOM_SEED,
            'BATCH_SIZE':  BATCH_SIZE,
            'SHUFFLE_TRAIN':  SHUFFLE_TRAIN,
            'LEARNING_RATE': LEARNING_RATE,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'WINDOW_SIZE': WINDOW_SIZE,
        }
        experiments.write(dumps(info)+'\n')
    """
