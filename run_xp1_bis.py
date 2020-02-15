import os
from multiprocessing import Pool

if __name__ == "__main__":

################ EXPERIENCE 1 ####################

    # Une colonne = un dataset foong, boston, wine, concrete, kin8nm, yacht, powerplant
    
    # (max_iter, ensemble_size)
#    MAPs_params = [(200,1000) for i in range(7)]
    
    # (max_iter, learning_rate, patience)
    MFVIs_params = [(20000,0.05,500) for i in range(7)]
    
    # (max_iter, learning_rate, patience)
    GeNVIs_params = [(20000,0.05,500) for i in range(7)]

    datasets = [("foong", "0", MAPs_params[0], MFVIs_params[0], GeNVIs_params[0]), ("boston", "1", MAPs_params[1], MFVIs_params[1], GeNVIs_params[1]), ("wine", "0", MAPs_params[2], MFVIs_params[2], GeNVIs_params[2]), ("concrete", "1", MAPs_params[3], MFVIs_params[3], GeNVIs_params[3]), ("kin8nm", "0", MAPs_params[4], MFVIs_params[4], GeNVIs_params[4]), ("yacht", "1", MAPs_params[5], MFVIs_params[5], GeNVIs_params[5]), ("powerplant", "0", MAPs_params[6], MFVIs_params[6], GeNVIs_params[6])]
    
    
    MAPs = ["-m Experiments.MAP --max_iter=" + str(k[0]) + " --ensemble_size=" + str(k[1]) + " --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j,k,l,p in datasets]
    
    MFVIs = ["-m Experiments.MFVI --max_iter=" + str(l[0]) + " --init_std=1. --learning_rate=" + str(l[1]) + " --min_lr=0.000001 --patience=" + str(l[2]) + " --lr_decay=0.5 --n_ELBO_samples=100 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j,k,l,p in datasets]
    
    GeNVIs = ["-m Experiments.GeNVI --max_iter=" + str(p[0]) + " --learning_rate=" + str(p[1]) + " --patience=" + str(p[2]) + " --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j,k,l,p in datasets]
    
#    PTMCMCs = ["-m Experiments.PTMCMC --numiter=50 --burnin=1 --thinning=1 --temperatures=1.0,0.75,0.5,0.25,0.1 #--maintempindex=0 --baseMHproposalNoise=0.01 --temperatureNoiseReductionFactor=0.5 --std_init=1.0 --optimize=0 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j,k,l,p in datasets]
    
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
 
   
    pool = Pool(processes=2) 
    pool.map(run_dataset, MAPs[0:2])
    pool = Pool(processes=2) 
    pool.map(run_dataset, MAPs[2:4])  
    pool = Pool(processes=2) 
    pool.map(run_dataset, MAPs[4:6]) 
    pool = Pool(processes=2) 
    pool.map(run_dataset, MAPs[6:7])   

    pool = Pool(processes=2) 
    pool.map(run_dataset, MFVIs[0:2])  
    pool = Pool(processes=2) 
    pool.map(run_dataset, MFVIs[2:4]) 
    pool = Pool(processes=2) 
    pool.map(run_dataset, MFVIs[4:6])   
    pool = Pool(processes=2) 
    pool.map(run_dataset, MFVIs[6:7])   
    
    pool = Pool(processes=2) 
    pool.map(run_dataset, GeNVIs[0:2])  
    pool = Pool(processes=2) 
    pool.map(run_dataset, GeNVIs[2:4])   
    pool = Pool(processes=2) 
    pool.map(run_dataset, GeNVIs[4:6])   
    pool = Pool(processes=2) 
    pool.map(run_dataset, GeNVIs[6:7])   

################ EXPERIENCE 2 ####################

 # (max_iter, ensemble_size)
#    MAPs_params = [(1000,1000) for i in range(7)]
    
    # (max_iter, learning_rate, patience)
    MFVIs_params = [(20000,0.05,300) for i in range(7)]
    
    # (max_iter, learning_rate, patience)
    GeNVIs_params = [(20000,0.05,300) for i in range(7)]

    datasets = [("foong", "0", MAPs_params[0], MFVIs_params[0], GeNVIs_params[0]), ("boston", "1", MAPs_params[1], MFVIs_params[1], GeNVIs_params[1]), ("wine", "0", MAPs_params[2], MFVIs_params[2], GeNVIs_params[2]), ("concrete", "1", MAPs_params[3], MFVIs_params[3], GeNVIs_params[3]), ("kin8nm", "0", MAPs_params[4], MFVIs_params[4], GeNVIs_params[4]), ("yacht", "1", MAPs_params[5], MFVIs_params[5], GeNVIs_params[5]), ("powerplant", "0", MAPs_params[6], MFVIs_params[6], GeNVIs_params[6])]
    
    
    MAPs = ["-m Experiments.MAP --max_iter=" + str(k[0]) + " --ensemble_size=" + str(k[1]) + " --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j,k,l,p in datasets]
    
    MFVIs = ["-m Experiments.MFVI --max_iter=" + str(l[0]) + " --init_std=1. --learning_rate=" + str(l[1]) + " --min_lr=0.000001 --patience=" + str(l[2]) + " --lr_decay=0.5 --n_ELBO_samples=200 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j,k,l,p in datasets]
    
    GeNVIs = ["-m Experiments.GeNVI --max_iter=" + str(p[0]) + " --learning_rate=" + str(p[1]) + " --patience=" + str(p[2]) + " --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j,k,l,p in datasets]

    pool = Pool(processes=2) 
    pool.map(run_dataset, MAPs[0:2])
    pool = Pool(processes=2) 
    pool.map(run_dataset, MAPs[2:4])  
    pool = Pool(processes=2) 
    pool.map(run_dataset, MAPs[4:6]) 
    pool = Pool(processes=2) 
    pool.map(run_dataset, MAPs[6:7])   

    pool = Pool(processes=2) 
    pool.map(run_dataset, MFVIs[0:2])  
    pool = Pool(processes=2) 
    pool.map(run_dataset, MFVIs[2:4]) 
    pool = Pool(processes=2) 
    pool.map(run_dataset, MFVIs[4:6])   
    pool = Pool(processes=2) 
    pool.map(run_dataset, MFVIs[6:7])   
    
    pool = Pool(processes=2) 
    pool.map(run_dataset, GeNVIs[0:2])  
    pool = Pool(processes=2) 
    pool.map(run_dataset, GeNVIs[2:4])   
    pool = Pool(processes=2) 
    pool.map(run_dataset, GeNVIs[4:6])   
    pool = Pool(processes=2) 
    pool.map(run_dataset, GeNVIs[6:7])  