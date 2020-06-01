import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np

from Models import BigGenerator

from Experiments import draw_experiment, get_setup, save_model

import tempfile
device='cpu'

models=torch.load('Results/The_models.pt')
lat_dim=5
datasets=[d for d,i in models.items()]
methods=['GeNNeVI', 'FuNNeVI']

def run(dataset, method):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    
    G=BigGenerator(lat_dim, setup.param_count, device).to(device)

    log_device = 'cpu'
    
    RMSEs=[]
    LPPs=[]
    PICPs=[]
    MPIWs=[]
    
    for i,m in models[dataset][method].items():
        G.load_state_dict(m)
        theta=G(10000).detach()

        LPP_test, RMSE_test, _, PICP_test, MPIW_test = setup.evaluate_metrics(theta,log_device)
        
        RMSEs.append(RMSE_test[0].item())
        LPPs.append(LPP_test[0].item())
        PICPs.append(PICP_test.item())
        MPIWs.append(MPIW_test.item())
    
    metrics_dict={dataset:{'RMSE':(np.mean(RMSEs).numpy().round(decimals=3),np.std(RMSEs).numpy().round(decimals=3)),
                           'LPP': (np.mean(LPPs).numpy().round(decimals=3),np.std(LPPs).numpy().round(decimals=3)),
                           'PICP': np.mean(PICPs).numpy().round(decimals=3), 
                           'MPIW': np.mean(MPIWs).numpy().round(decimals=3), 
                          }
                 }
    
    return metrics_dict


if __name__ == "__main__":

    
    results={}

    for m in methods:
        for d in datasets:
            metrics=run(d, m) 
            print(d+': done :-)')
            results.update(metrics)

    torch.save(results, 'Results/MR_metrics.pt')
