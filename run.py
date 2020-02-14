import os
from multiprocessing import Pool

if __name__ == "__main__":
    
    datasets = [("foong", "0"), ("boston", "1"), ("wine", "0"), ("concrete", "1"), ("california","0")]
    
#    datasets_1 = [("foong", "0"), ("boston", "1"), ("wine", "0")]
#    datasets_2 = [("concrete", "1"), ("california","0")]
    
    MAPs = ["-m Experiments.MAP --max_iter=10 --ensemble_size=1 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j in datasets]
    
    MFVIs = ["-m Experiments.MFVI --max_iter=10 --init_std=1. --learning_rate=0.1 --min_lr=0.000001 --patience=100 --lr_decay=0.5 --n_ELBO_samples=100 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j in datasets]
    
    GeNVIs = ["-m Experiments.GeNVI --max_iter=50 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j in datasets]
    
    PTMCMCs = ["-m Experiments.PTMCMC --numiter=50 --burnin=1 --thinning=1 --temperatures=1.0,0.75,0.5,0.25,0.1 #--maintempindex=0 --baseMHproposalNoise=0.01 --temperatureNoiseReductionFactor=0.5 --std_init=1.0 --optimize=0 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j in datasets]
    
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
                                                                                
    pool = Pool(processes=len(datasets_1)) 
    
    pool.map(run_dataset, MAPs)
#    pool.map(run_dataset, MFVIs)   
#    pool.map(run_dataset, GeNVIs)   
#    pool.map(run_dataset, PTMCMCs)   
   