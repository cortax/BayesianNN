import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np


from Inference.FuNNeVI import FuNNeVI
from Models import BigGenerator, SLPGenerator
from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model

import tempfile

lat_dim=5
nb_models=3#to change
NNE=1
ratio_ood=.1
p_norm=2
n_samples_KL=500
n_samples_LL=100
max_iter=20000
learning_rate=0.005
patience=1000
min_lr= 0.001
lr_decay=.7
device='cuda:0'

def learning(loglikelihood, batch, size_data, prior, projection, n_samples_FU, ratio_ood, p,
                   lat_dim, param_count, 
                   kNNE, n_samples_KL, n_samples_LL,  
                   max_iter, learning_rate, min_lr, patience, lr_decay, 
                   device, rho):

    GeN = BigGenerator(lat_dim, param_count,device).to(device)
    #GeN=SLPGenerator(lat_dim, param_count,device).to(device)
    #GeN = GeNetEns(ensemble_size, lat_dim, layerwidth, param_count, activation, init_w, init_b, device)

    optimizer = FuNNeVI(loglikelihood, batch, size_data, prior, projection, n_samples_FU, ratio_ood, p,
                          kNNE, n_samples_KL, n_samples_LL, 
                          max_iter, learning_rate, min_lr, patience, lr_decay,
                          device, rho=rho)

    ELBO = optimizer.run(GeN)

    return GeN, optimizer.scores, ELBO.item()



def log_GeNVI_experiment(setup,  n_samples_FU, ratio_ood, p, batch,
                         lat_dim, 
                         kNNE, n_samples_KL, n_samples_LL, 
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device):
    
    mlflow.set_tag('batch_size', batch)

        
    mlflow.set_tag('sigma_noise', setup.sigma_noise)    

    mlflow.set_tag('sigma_prior', setup.sigma_prior)    
    mlflow.set_tag('device', device)
    mlflow.set_tag('param_dim', setup.param_count)
    mlflow.set_tag('NNE', kNNE)
   
    mlflow.set_tag('device', device)

    mlflow.log_param('lat_dim', lat_dim)
    
    mlflow.log_param('L_p norm', p)

    mlflow.log_param('n_samples_FU', n_samples_FU)
    mlflow.log_param('ratio_ood', ratio_ood)
    mlflow.log_param('n_samples_KL', n_samples_KL)
    mlflow.log_param('n_samples_LL', n_samples_LL)
    

    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('patience', patience)
    mlflow.log_param('lr_decay', lr_decay)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('min_lr', min_lr)

def log_GeNVI_run(ELBO, scores):    

    mlflow.log_metric("The elbo", float(ELBO))



    for t in range(len(scores['ELBO'])):
        mlflow.log_metric("elbo", float(scores['ELBO'][t]), step=100*t)
        mlflow.log_metric("KL", float(scores['KL'][t]), step=100*t)
        mlflow.log_metric("LL", float(scores['LL'][t]), step=100*t)        
        mlflow.log_metric("learning_rate", float(scores['lr'][t]), step=100*t)
        


def run(setup, n_samples_FU):
    
    setup_ = get_setup( setup)
    setup=setup_.Setup( device) 
    
    loglikelihood=setup.loglikelihood
    projection=setup.projection
    size_sample=setup.n_train_samples
    param_count=setup.param_count
    input_dim=setup.input_dim

    batch=np.min([int(size_sample/6),500])

    rho=batch/size_sample * input_dim
    
    ratio_ood=0.05

    if dataset== 'concrete':
        ratio_ood=0.25
    
    if dataset == 'boston':
        ratio_ood=0.1
    if dataset == 'yacht':
        ratio_ood=0.2
    
    
    
    if dataset== 'wine':
        ratio_ood=0.00
    

        
        
    def prior(n):
        return setup.sigma_prior*torch.randn(size=(n,param_count), device= device)


    
    
    xpname = setup.experiment_name + '/FuNNeVI-mres'
    mlflow.set_experiment(xpname)
    
    with mlflow.start_run():

        log_GeNVI_experiment(setup,  n_samples_FU,  ratio_ood,  p_norm, batch,
                              lat_dim, 
                              NNE,  n_samples_KL,  n_samples_LL,
                              max_iter,  learning_rate,  min_lr,  patience,  lr_decay,
                              device)
        
        GeN_models_dict=[]
        for i in range( nb_models):
            with mlflow.start_run(run_name=str(i),nested=True):
                start = timeit.default_timer()
    
                GeN, log_scores, ELBO = learning(loglikelihood, batch, setup.n_train_samples,
                                                                        prior, projection, 
                                                                         n_samples_FU,  ratio_ood,  p_norm,
                                                                         lat_dim, setup.param_count,
                                                                         NNE,  n_samples_KL,  n_samples_LL,
                                                                         max_iter,  learning_rate,  min_lr,  patience,
                                                                         lr_decay,  device, rho=rho)


                stop = timeit.default_timer()
                execution_time = stop - start

                log_GeNVI_run(ELBO, log_scores)
                """
                log_device = 'cpu'
                theta = GeN(1000).detach().to(log_device)
                log_exp_metrics(setup.evaluate_metrics, theta, execution_time, log_device)
                """
                save_model(GeN)
                GeN_models_dict.append((i,GeN.state_dict().copy()))
                
                if setup.plot:
                    log_device = 'cpu'
                    theta = GeN(1000).detach().to(log_device)
                    draw_experiment(setup, theta[0:1000], log_device)
      
        tempdir = tempfile.TemporaryDirectory()
        models={str(i): model for i,model in GeN_models_dict}
        torch.save(models, tempdir.name + '/models.pt')
        mlflow.log_artifact(tempdir.name + '/models.pt')
    
    return models


models=torch.load('Results/FuNmodelsPatience6pm.pt')#torch.load('Results/The_models.pt')
lat_dim=5
datasets=[d for d,i in models.items()]
methods=['FuNNeVI']#['GeNNeVI', 'FuNNeVI']

def run_metrics(dataset, method):
    
    log_device = 'cpu'
    device='cuda:0'
    setup_ = get_setup(dataset)
    setup=setup_.Setup(log_device) 
    
    G=BigGenerator(lat_dim, setup.param_count, device).to(device)

    
    
    RMSEs=[]
    LPPs=[]
    PICPs=[]
    MPIWs=[]
    
    for i,m in models[dataset].items():
        G.load_state_dict(m)
        n=10000
        if dataset == 'powerplant':
            n=2000
        theta=G(n).detach().cpu()

        LPP_test, RMSE_test, _, PICP_test, MPIW_test = setup.evaluate_metrics(theta,log_device)
        
        RMSEs.append(RMSE_test[0].item())
        LPPs.append(LPP_test[0].item())
        PICPs.append(PICP_test.item())
        MPIWs.append(MPIW_test.item())
    
    metrics_dict={(method,dataset):{'RMSE':(np.mean(RMSEs).round(decimals=3),np.std(RMSEs).round(decimals=3)),
                           'LPP': (np.mean(LPPs).round(decimals=3),np.std(LPPs).round(decimals=3)),
                           'PICP':  (np.mean(PICPs).round(decimals=3),np.std(PICPs).round(decimals=3)), 
                           'MPIW':  (np.mean(MPIWs).round(decimals=3),np.std(MPIWs).round(decimals=3))
                           }
                   }
                 
    
    return metrics_dict



if __name__ == "__main__":
    
    
    FuNmodels={}
    n_samples_FU=200
    
    for dataset in ['powerplant','boston', 'yacht', 'concrete','energy', 'wine']:
        print(dataset)
        models=run(dataset, n_samples_FU=n_samples_FU) 
        print(dataset+': done :-)')
        FuNmodels.update({dataset:models})
        torch.save(FuNmodels, 'Results/FuNmodelsLast.pt')

    
    models=FuNmodels
    lat_dim=5
    datasets=[d for d,i in models.items()]
    methods=['FuNNeVI']#['GeNNeVI', 'FuNNeVI']
    
       
    results={}

    for m in methods:
        for d in datasets:
            metrics=run_metrics(d, m) 
            print(d+': done :-)')
            results.update(metrics)

    torch.save(results, 'Results/MR_FuNmetrics_last.pt')



