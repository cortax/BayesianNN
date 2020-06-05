import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np

from tempfile import TemporaryDirectory

from Inference.GeNNeVI import GeNNeVI
from Models import BigGenerator
from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model

import tempfile

lat_dim=5
nb_models=3
NNE=1
n_samples_KL=500
n_samples_LL=100
max_iter=30000
learning_rate=0.005
patience=1000
min_lr= 0.001
lr_decay=.7
device='cuda:0'

def learning(loglikelihood, batch, size_data, prior,
                   lat_dim, param_count, 
                   kNNE, n_samples_KL, n_samples_LL,  
                   max_iter, learning_rate, min_lr, patience, lr_decay, 
                   device):

    GeN = BigGenerator(lat_dim, param_count,device).to(device)
    #GeN=SLPGenerator(lat_dim, param_count,device).to(device)
    #GeN = GeNetEns(ensemble_size, lat_dim, layerwidth, param_count, activation, init_w, init_b, device)

    optimizer = GeNNeVI(loglikelihood, batch, size_data, prior,
                          kNNE, n_samples_KL, n_samples_LL, 
                          max_iter, learning_rate, min_lr, patience, lr_decay,
                          device)

    ELBO = optimizer.run(GeN)

    return GeN, optimizer.scores, ELBO.item()





def log_GeNVI_experiment(setup,  batch,
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
        


def run(setup, batch=None):
    
    setup_ = get_setup( setup)
    setup=setup_.Setup( device) 
    
    loglikelihood=setup.loglikelihood
    projection=setup.projection
    size_sample=setup.n_train_samples
    param_count=setup.param_count

    
    batch= batch
    if batch is None:
        batch=int(size_sample/6)
    
    

    def prior(n):
        return setup.sigma_prior*torch.randn(size=(n,param_count), device= device)


    
    xpname = setup.experiment_name + '/GeNNeVI-mr'
    mlflow.set_experiment(xpname)
    
    with mlflow.start_run():

        log_GeNVI_experiment(setup, batch,
                              lat_dim, 
                              NNE,  n_samples_KL,  n_samples_LL,
                              max_iter,  learning_rate,  min_lr,  patience,  lr_decay,
                              device)


        RMSEs=[]
        LPPs=[]
        TIMEs=[]

        GeN_models_dict=[]
        
        for i in range( nb_models):

            setup=setup_.Setup( device) 

            loglikelihood=setup.loglikelihood
            projection=setup.projection
            size_sample=setup.n_train_samples
            param_count=setup.param_count

            with mlflow.start_run(run_name=str(i),nested=True):
                start = timeit.default_timer()

                GeN, log_scores, ELBO = learning(loglikelihood, batch, setup.n_train_samples, prior, 
                                                  lat_dim, setup.param_count,
                                                  NNE,  n_samples_KL,  n_samples_LL,
                                                  max_iter,  learning_rate,  min_lr,  patience,
                                                  lr_decay,  device)


                stop = timeit.default_timer()
                execution_time = stop - start

                log_GeNVI_run(ELBO, log_scores)

                log_device = 'cpu'
                theta = GeN(1000).detach().to(log_device)
                LPP_test, RMSE_test, _ =log_exp_metrics(setup.evaluate_metrics, theta, execution_time, log_device)
                RMSEs.append(RMSE_test[0])
                LPPs.append(LPP_test[0])
                TIMEs.append(execution_time)
                save_model(GeN)
                GeN_models_dict.append((i,GeN.state_dict().copy()))
                
                if setup.plot:
                    draw_experiment(setup, theta[0:1000], log_device)


        mlflow.log_metric('average time',np.mean(TIMEs))

        mlflow.log_metric('RMSE',np.mean(RMSEs))
        mlflow.log_metric('RMSE-std',np.std(RMSEs))
        mlflow.log_metric('LPP',np.mean(LPPs))
        mlflow.log_metric('LPP-std',np.std(LPPs))


        tempdir = tempfile.TemporaryDirectory()
        models={str(i): model for i,model in GeN_models_dict}
        torch.save(models, tempdir.name + '/models.pt')
        mlflow.log_artifact(tempdir.name + '/models.pt')

    return models



if __name__ == "__main__":
    
    GeNmodels={}


    dataset='powerplant'
    print(dataset)
    models=run(dataset, batch=500) 
    print(dataset+': done :-)')
    
    GeNmodels.update({dataset:models})
    
    for dataset in ['boston', 'yacht', 'concrete','energy', 'wine']:
        print(dataset)
        models=run(dataset) 
        print(dataset+': done :-)')
        GeNmodels.update({dataset:models})



    for dataset in ['foong','foong_mixed', 'foong_sparse']:
        print(dataset)
        models=run(dataset) 
        print(dataset+': done :-)')
        GeNmodels.update({dataset:models})



    torch.save(GeNmodels, 'Results/GeNmodels_minibatch.pt')

  
