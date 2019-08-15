import argparse

from source.experiments.train_hn_gru_net import *


parser = argparse.ArgumentParser()

"""
Required
"""
#params for redux
parser.add_argument('-exp', '--EXPERIMENT_NUMBER', action='store', nargs='?',
                    default=0, required=True, type=int)
parser.add_argument('-m', '--MODULE', action='store', nargs='?',
                    default='model', required=True, type=str)
parser.add_argument('-mm', '--MODULE_MODE', action='store', nargs='?',
                    default='train', required=True, type=str)
parser.add_argument('-ml', '--MODEL', action='store', nargs='?',
                    default='hn_gru_net', required=True, type=str)

"""
Non-Required
"""
#params for restarts
parser.add_argument('-r', '--RESTART', action='store', nargs='?',
                    default=False, required=False, type=bool)
parser.add_argument('-rmws', '--RESTART_MODEL_WEIGHTS_STRING', action='store', nargs='?',
                    default=None, required=False, type=str)

#params for dataset
parser.add_argument('--FORECASTING', action='store', nargs='?',
                    default=True, required=False, type=bool)
parser.add_argument('--FEATURE_ENDO', action='store', nargs='?',
                    default=True, required=False, type=bool)
parser.add_argument('--FEATURE_EXO', action='store', nargs='?',
                    default=True, required=False, type=bool)
parser.add_argument('-sw', '--WINDOWS_SIZE', action='store', nargs='?', 
                    default=8, required=False, type=int)
parser.add_argument('-target', '--TARGET_CHOICE', action='store', nargs='?', 
                    default=1, required=False, type=int)
parser.add_argument('-delta', '--SHIFT_DELTA', action='store', nargs='?', 
                    default=1, required=False, type=int)

#params for dataloader
parser.add_argument('-ttr', '--TRAIN_TEST_RATIO', action='store', nargs='?', 
                    default=0.8, required=False, type=float)
parser.add_argument('-tvr', '--TRAIN_VALID_RATIO', action='store', nargs='?', 
                    default=0.8, required=False, type=float)
parser.add_argument('-st', '--SAMPLER_TYPE', action='store', nargs='?',
                    default=None, required=False, type=str)

#params for RNN model
parser.add_argument('-hs', '--HIDDEN_SIZE', action='store', nargs='?',
                    default=128, required=False, type=int)
parser.add_argument('-nl', '--NUM_LAYERS', action='store', nargs='?',
                    default=1, required=False, type=int)

#params for RNN dropout
parser.add_argument('--DROPOUT_HIDDEN', action='store', nargs='?', 
                    default=0.0, required=False, type=float)
parser.add_argument('--DROPOUT_HN', action='store', nargs='?', 
                    default=0.0, required=False, type=float)

#params for seed and other hardware constraints
parser.add_argument('--RANDOM_SEED', action='store', nargs='?',
                    default=42, required=False, type=int)
parser.add_argument('-nw', '--NUM_WORKERS', action='store', nargs='?',
                    default=8, required=False, type=int)
parser.add_argument('--USE_GPU', action='store', nargs='?',
                    default=True, required=False, type=bool)
parser.add_argument('--GPU_DEVICE_ID', action='store', nargs='?',
                    default=0, required=False, type=int)

#params for optim and scheduler
parser.add_argument('-bz', '--BATCH_SIZE', action='store', nargs='?', 
                    default=32, required=False, type=int)
parser.add_argument('-e', '--EPOCHS', action='store', nargs='?',
                    default=500, required=False, type=int)
parser.add_argument('-optim', '--OPTIM_TYPE', action='store', nargs='?', 
                    default='adam', required=False, type=str)
parser.add_argument('--EARLY_STOP', action='store', nargs='?', 
                    default=100, required=False, type=int)
parser.add_argument('--PATIENCE', action='store', nargs='?', 
                    default=50, required=False, type=int)
parser.add_argument('--COOLDOWN', action='store', nargs='?',
                    default=0, required=False, type=int)
parser.add_argument('--FACTOR', action='store', nargs='?', 
                    default=0.8, required=False, type=float)
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
    print('TODO.... Maybe')

def recall_test():
    print('TODO')

if __name__ == '__main__':
    #TODO: The comments on what arg is used by what module
    args = parser.parse_args()

    print('\n----------ARGUMENTS----------\n')
    print(args, '\n')
    print('\n----------MODULE START----------\n')

    if args.MODULE=='debug':
        print('debug')
        debug()
    elif args.MODULE=='recall':
        print('Recall Test')
        recall_test()
    elif args.MODULE=='model':
        print('model')
        if args.MODULE_MODE=='train':
            if args.MODEL=='hn_gru_net':
                hn_gru_net_main(
                    args.EXPERIMENT_NUMBER, 
                    args.EPOCHS, args.BATCH_SIZE, args.SAMPLER_TYPE,
                    args.EARLY_STOP, args.PATIENCE, args.COOLDOWN, args.FACTOR, args.LEARNING_RATE, args.VERBOSE, args.THRESHOLD,
                    args.FORECASTING, args.FEATURE_ENDO, args.FEATURE_EXO, args.TARGET_CHOICE, 
                    args.WINDOWS_SIZE, args.SHIFT_DELTA,
                    args.TRAIN_TEST_RATIO, args.TRAIN_VALID_RATIO,
                    args.OPTIM_TYPE, args.WEIGHT_DECAY, args.MOMENTUM, args.DAMPENING,
                    args.HIDDEN_SIZE, args.NUM_LAYERS, args.DROPOUT_HIDDEN, args.DROPOUT_HN,
                    args.RANDOM_SEED, args.NUM_WORKERS, args.USE_GPU, args.GPU_DEVICE_ID, 
                    args.RESTART, args.RESTART_MODEL_WEIGHTS_STRING
                    )
