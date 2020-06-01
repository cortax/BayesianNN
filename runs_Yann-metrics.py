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
        theta=G(1000).detach()

        LPP_test, RMSE_test, _, PICP_test, MPIW_test = setup.evaluate_metrics(theta,log_device)
        
        RMSEs.append(RMSE_test[0].item())
        LPPs.append(LPP_test[0].item())
        PICPs.append(PICP_test.item())
        MPIWs.append(MPIW_test.item())
    
    metrics_dict={{dataset:{(method,'RMSE'):(np.mean(RMSEs).round(decimals=3),np.std(RMSEs).round(decimals=3)),
                           (method,'LPP'): (np.mean(LPPs).round(decimals=3),np.std(LPPs).round(decimals=3)),
                           (method,'PICP'): {dataset: np.mean(PICPs).round(decimals=3), 
                           (method,'MPIW'): {dataset: np.mean(MPIWs).round(decimals=3), 
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
