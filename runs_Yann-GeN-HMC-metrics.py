import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np


from Experiments import draw_experiment, get_setup, save_model

import tempfile
device='cpu'

models_HMC=torch.load('Results/models_HMC.pt')

datasets=[d for d,i in models_HMC.items()]

def run(dataset):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    

    log_device = 'cpu'
    theta_ = models_HMC[dataset]
    theta = theta_
    
    LPP_test, RMSE_test, _, PICP_test, MPIW_test = setup.evaluate_metrics(theta,log_device)
         
    metrics_dict={dataset:{'RMSE':RMSE_test[0].numpy().round(decimals=3),
                           'LPP': LPP_test[0].numpy().round(decimals=3),
                           'PICP': PICP_test.numpy().round(decimals=3), 
                           'MPIW': MPIW_test.numpy().round(decimals=3), 
                          }
                 }
    
    return metrics_dict


if __name__ == "__main__":

    
    results={}

    
    for d in datasets:
        metrics=run(d) 
        print(d+': done :-)')
        results.update(metrics)

    torch.save(results, 'Results/HMC_metrics.pt')
