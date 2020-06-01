import torch
from torch import nn
import argparse
import mlflow
import timeit



from Inference.FuNNeVI import FuNNeVI
from Models import BigGenerator, SLPGenerator
from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model

import tempfile

lat_dim=5
nb_models=1
NNE=1
ratio_ood=1.
p_norm=2
n_samples_KL=100
n_samples_LL=100
max_iter=20000
learning_rate=0.005
patience=2000
min_lr= 0.001
lr_decay=.7
device='cuda:0'

def learning(loglikelihood, batch, size_data, prior, projection, n_samples_FU, ratio_ood, p,
                   lat_dim, param_count, 
                   kNNE, n_samples_KL, n_samples_LL,  
                   max_iter, learning_rate, min_lr, patience, lr_decay, 
                   device):

    GeN = BigGenerator(lat_dim, param_count,device).to(device)
    #GeN=SLPGenerator(lat_dim, param_count,device).to(device)
    #GeN = GeNetEns(ensemble_size, lat_dim, layerwidth, param_count, activation, init_w, init_b, device)

    optimizer = FuNNeVI(loglikelihood, batch, size_data, prior, projection, n_samples_FU, ratio_ood, p,
                          kNNE, n_samples_KL, n_samples_LL, 
                          max_iter, learning_rate, min_lr, patience, lr_decay,
                          device)

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

    batch=int(size_sample/10)

    
    

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
                                                                         lr_decay,  device)


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


if __name__ == "__main__":
    
    FuNmodels={}
    n_samples_FU=500
    
    for dataset in ['powerplant','boston', 'yacht', 'concrete','energy', 'wine']:
        print(dataset)
        models=run(dataset, n_samples_FU=n_samples_FU) 
        print(dataset+': done :-)')
        FuNmodels.update({dataset:models})
        torch.save(FuNmodels, 'Results/FuNmodelsPatience5pm.pt')




    """  
    print('kin8nm')
    pool.map(run_dataset, ["-m Experiments.FuNNeVI-mr --batch=100 --nb_models="+str(n)+" --setup=kin8nm --NNE=10  --device='cuda:0'"])
    pool.map(run_dataset, ["-m Experiments.GeNNeVI-mr --batch=100 --nb_models="+str(n)+" --setup=kin8nm --device='cuda:0'"])      
    print('kin8nm: done :-)')
   
  
          #sort of early stopping for FuNNeVI
    pool = Pool(processes=1) 

    for dataset in ['boston', 'yacht', 'concrete','energy', 'wine','powerplant']:
        print(dataset)
        pool.map(run_dataset, ["-m Experiments.FuNNeVI-mres --batch=100  --nb_models="+str(n)+" --setup="+dataset+"  --device='cuda:0'"])  
        print(dataset+': done :-)')
    
    print('kin8nm')
    pool.map(run_dataset, ["-m Experiments.FuNNeVI-mres --batch=100  --nb_models="+str(n)+" --setup=kin8nm --NNE=10  --device='cuda:0'"])

    print('kin8nm: done :-)')
    
   
    for dataset in ['foong','foong_mixed', 'foong_sparse']:
        print(dataset)
        pool.map(run_dataset, ["-m Experiments.FuNNeVI-mres --n_samples_FU=20 --nb_models="+str(n)+" --setup="+dataset+"  --device='cuda:0'"])  

        print(dataset+': done :-)')
        
           #sort of early stopping for FuNNeVI
    pool = Pool(processes=1) 

    for dataset in ['boston', 'yacht', 'concrete','energy', 'wine','powerplant']:
        print(dataset)
        pool.map(run_dataset, ["-m Experiments.FuNNeVI-spes --setup="+dataset+"  --device='cuda:0'"])  
        print(dataset+': done :-)')
    
    print('kin8nm')
    pool.map(run_dataset, ["-m Experiments.FuNNeVI-spes --setup=kin8nm --NNE=1  --device='cuda:0'"])

    print('kin8nm: done :-)')
    
    """
