import argparse

from source.experiments.train_hn_gru_net import *
from source.utils.inference import *


parser = argparse.ArgumentParser()

"""
Required
"""
#params for redux
parser.add_argument('-exp', '--EXPERIMENT_NUMBER', action='store',
                    default=0, required=True, type=int)
parser.add_argument('-m', '--MODULE', action='store',
                    default='model', required=True, type=str)
parser.add_argument('-mm', '--MODULE_MODE', action='store',
                    default='train', required=True, type=str)
parser.add_argument('-ml', '--MODEL', action='store',
                    default='hn_gru_net', required=True, type=str)

"""
Non-Required
"""
#params for restarts
parser.add_argument('-r', '--RESTART', action='store_true', required=False)
parser.add_argument('-rm', '--RESTART_MODEL', action='store',
                    default=None, required=False, type=str)

#params for dataset
parser.add_argument('--NOT_FORECASTING', action='store_false', required=False)
parser.add_argument('--NO_FEATURE_ENDO', action='store_false', required=False)
parser.add_argument('--NO_FEATURE_EXO', action='store_false', required=False)
parser.add_argument('-ws', '--WINDOWS_SIZE', action='store', 
                    default=8, required=False, type=int)
parser.add_argument('-target', '--TARGET_CHOICE', action='store', 
                    default=1, required=False, type=int)
parser.add_argument('-delta', '--SHIFT_DELTA', action='store', 
                    default=1, required=False, type=int)
parser.add_argument('-stride', '--TIME_STRIDE', action='store', 
                    default=1, required=False, type=int)
                    
#params for dataloader
parser.add_argument('-ttr', '--TRAIN_TEST_RATIO', action='store', 
                    default=0.8, required=False, type=float)
parser.add_argument('-tvr', '--TRAIN_VALID_RATIO', action='store', 
                    default=0.8, required=False, type=float)
parser.add_argument('-st', '--SAMPLER_TYPE', action='store',
                    default=None, required=False, type=str)

#params for RNN model
parser.add_argument('-hs', '--HIDDEN_SIZE', action='store',
                    default=64, required=False, type=int)
parser.add_argument('-nl', '--NUM_LAYERS', action='store',
                    default=1, required=False, type=int)

#params for RNN dropout
parser.add_argument('--DROPOUT_HIDDEN', action='store', 
                    default=0.0, required=False, type=float)
parser.add_argument('--DROPOUT_HN', action='store', 
                    default=0.0, required=False, type=float)

#params for seed and other hardware constraints
parser.add_argument('--RANDOM_SEED', action='store',
                    default=42, required=False, type=int)
parser.add_argument('-nw', '--NUM_WORKERS', action='store',
                    default=4, required=False, type=int)
parser.add_argument('--NO_GPU', action='store_false', required=False)
parser.add_argument('--GPU_DEVICE_ID', action='store',
                    default=0, required=False, type=int)

#params for optim and scheduler
parser.add_argument('-bz', '--BATCH_SIZE', action='store', 
                    default=32, required=False, type=int)
parser.add_argument('-e', '--EPOCHS', action='store',
                    default=500, required=False, type=int)
parser.add_argument('-optim', '--OPTIM_TYPE', action='store', 
                    default='adam', required=False, type=str)
parser.add_argument('-es', '--EARLY_STOP', action='store', 
                    default=100, required=False, type=int)
parser.add_argument('--PATIENCE', action='store', 
                    default=50, required=False, type=int)
parser.add_argument('--COOLDOWN', action='store',
                    default=0, required=False, type=int)
parser.add_argument('--FACTOR', action='store', 
                    default=0.8, required=False, type=float)
parser.add_argument('-lr', '--LEARNING_RATE', action='store',
                    default=0.02, required=False, type=float)                    
parser.add_argument('--NO_VERBOSE', action='store_false', required=False)
parser.add_argument('--THRESHOLD', action='store',
                    default=1e-6, required=False, type=float)
parser.add_argument('--WEIGHT_DECAY', action='store', 
                    default=0.0, required=False, type=float)
parser.add_argument('--MOMENTUM', action='store', 
                    default=0.9, required=False, type=float)
parser.add_argument('--DAMPENING', action='store', 
                    default=0.0, required=False, type=float)

def debug():
    print('TODO... Maybe')

if __name__ == '__main__':
    #TODO: The comments on what arg is used by what module
    args = parser.parse_args()

    print('\n----------ARGUMENTS----------\n')
    print(args, '\n')
    print('\n----------MODULE START----------\n')

    if args.MODULE=='debug':
        print('debug')
        debug()
    elif args.MODULE=='model':
        print('model')
        if args.MODULE_MODE=='train':
            if args.MODEL=='hn_gru_net':
                hn_gru_net_main(
                    args.EXPERIMENT_NUMBER, 
                    args.EPOCHS, args.BATCH_SIZE, args.SAMPLER_TYPE,
                    args.EARLY_STOP, args.PATIENCE, args.COOLDOWN, args.FACTOR, args.LEARNING_RATE, args.NO_VERBOSE, args.THRESHOLD,
                    args.NOT_FORECASTING, args.NO_FEATURE_ENDO, args.NO_FEATURE_EXO, args.TARGET_CHOICE, 
                    args.WINDOWS_SIZE, args.SHIFT_DELTA, args.TIME_STRIDE,
                    args.TRAIN_TEST_RATIO, args.TRAIN_VALID_RATIO,
                    args.OPTIM_TYPE, args.WEIGHT_DECAY, args.MOMENTUM, args.DAMPENING,
                    args.HIDDEN_SIZE, args.NUM_LAYERS, args.DROPOUT_HIDDEN, args.DROPOUT_HN,
                    args.RANDOM_SEED, args.NUM_WORKERS, args.NO_GPU, args.GPU_DEVICE_ID, 
                    args.RESTART, args.RESTART_MODEL
                    )
