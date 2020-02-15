import os
from multiprocessing import Pool

if __name__ == "__main__":

################ EXPERIENCE 1 ####################

    # Une colonne = un dataset foong, boston, wine, concrete, kin8nm, yacht, powerplant
    
    # (numiter, baseMHproposalNoise)
    PTMCMCs_params = [(10000, 0.002) for i in range(7)]
    
    datasets = [("foong", PTMCMCs_params[0]), ("boston", PTMCMCs_params[1]), ("wine",PTMCMCs_params[2]), ("concrete", PTMCMCs_params[3]), ("kin8nm", PTMCMCs_params[4]), ("yacht", PTMCMCs_params[5]), ("powerplant", PTMCMCs_params[6])]
    

    PTMCMCs = ["-m Experiments.PTMCMC --numiter=" + str(j[0]) +  " --baseMHproposalNoise=" + str(j[1]) + " --setup=" + i for i,j in datasets]
    
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
 
      
    pool = Pool(processes=1) 
    pool.map(run_dataset, PTMCMCs[0:1])  
    pool = Pool(processes=1) 
    pool.map(run_dataset, PTMCMCs[1:2])   
    pool = Pool(processes=1) 
    pool.map(run_dataset, PTMCMCs[2:3])   
    pool = Pool(processes=1) 
    pool.map(run_dataset, PTMCMCs[3:4])   
    pool = Pool(processes=1) 
    pool.map(run_dataset, PTMCMCs[4:5])  
    pool = Pool(processes=1) 
    pool.map(run_dataset, PTMCMCs[5:6])   
    pool = Pool(processes=1) 
    pool.map(run_dataset, PTMCMCs[6:7])   


################ EXPERIENCE 2 ####################

