import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np

from Models import BigGenerator

from Experiments import draw_experiment, get_setup, save_model
from Tools import KL, batchKL, sw, FunSW, FunKL

import tempfile
device='cpu'

lat_dim=5

def _FunKL(s,t,projection,device):
    k=1
    FKL=FunKL(s,t,projection=projection,device=device,k=k)
    while torch.isnan(FKL):
        k+=1
        FKL=FunKL(s,t,projection=projection,device=device,k=k)
    return FKL
    

models_HMC=torch.load('Results/models_HMC.pt')

models=torch.load('Results/GeNmodels.pt')
datasets=[d for d,m in models_HMC.items()]

methods=['GeNNeVI']
ratio_ood=1.

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

metrics={}




def run(dataset, method, metrics):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    G=BigGenerator(lat_dim, setup.param_count, device).to(device)
    def projection(t,s,m):
        return setup.projection(t,s,n_samples=m,ratio_ood=ratio_ood)

    
    metric='KL(-,HMC)'
    KLs=[]
    for i,m in models[dataset].items():
        t=models_HMC[dataset].to(device)
        G.load_state_dict(m)
        s=G(t.shape[0]).detach()
        K=_FunKL(s,t,projection,device)
        print(K)
        KLs.append(K.item())
    print(KLs)

    metrics.update({(method,dataset):{metric:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))}})
    
    metric='KL(HMC,-)'
    KLs=[]
    for i,m in models[dataset].items():
        t=models_HMC[dataset].to(device)
        G.load_state_dict(m)
        s=G(t.shape[0]).detach()
        K=_FunKL(t,s,projection,device)
        print(dataset+': '+str(K.item()))
        KLs.append(K.item())
    print(KLs)
    
    metrics.update({(method,dataset):{metric:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))}})
    
    metric='SW(-,G)'
    KLs=[]
    for i,m in models[dataset].items():
        t=models_HMC[dataset].to(device)
        G.load_state_dict(m)
        s=G(t.shape[0]).detach()
        K=FunSW(t,s, projection, device)
        print(dataset+': '+str(K.item()))
        KLs.append(K.item())
    print(KLs)
    metrics.update({(method,dataset):{metric:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))}})
    
    metric='KL(-,-)'
    KLs=[]
    for (i,m),(j,n) in models_pairs:
        G.load_state_dict(m)
        s=G(10000).detach()
        G.load_state_dict(n)
        t=G(10000).detach()
        K=_FunKL(t,s, projection,device)
        K_=_FunKL(s,t, projection,device)
        print(dataset+': '+str((K.item(), K_.item())))
        KLs.append(K.item())
        KLs.append(K_.item())
    print(KLs)
    metrics.update({(method,dataset):{metric:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))}})
    
    metric='SW(-,-)'
    KLs=[]
    for (i,m),(j,n) in models_pairs:
        G.load_state_dict(m)
        s=G(10000).detach()
        G.load_state_dict(n)
        t=G(10000).detach()
        SW=sw(s.cpu(),t.cpu(),'cpu')
        print(SW)
        KLs.append(SW.item())
    
    metrics.update({(method,dataset):{metric:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))}})

    return metrics


if __name__ == "__main__":
    
    
    metrics={}
    
    
    for m in methods:
        for d in datasets:
            metrics=run(d, m, metrics) 
            print(d+': done :-)')

    torch.save(results, 'Results/DivergenceGeN.pt')
