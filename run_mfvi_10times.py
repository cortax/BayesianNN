import os
from multiprocessing import Pool

if __name__ == "__main__":

   
    # (datasets, GPU)
    datasets = [("foong", "0"), ("boston", "0"), ("wine", "0"), ("concrete", "0"), ("kin8nm", "0"), ("yacht", "0"), ("powerplant", "0")]

    MFVIs = ["-m Experiments.MFVI --max_iter=20000 --learning_rate=0.05 --n_ELBO_samples=10 --patience=500 --lr_decay=.5 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j in datasets]
    
    
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
    
    for i in range(0,10):
        pool = Pool(processes=1) 
        pool.map(run_dataset, MFVIs[0:1])  
        pool = Pool(processes=1) 
        pool.map(run_dataset, MFVIs[1:2])   
        pool = Pool(processes=1) 
        pool.map(run_dataset, MFVIs[2:3])   
        pool = Pool(processes=1) 
        pool.map(run_dataset, MFVIs[3:4])   
        pool = Pool(processes=1) 
        pool.map(run_dataset, MFVIs[4:5])   
        pool = Pool(processes=1) 
        pool.map(run_dataset, MFVIs[5:6])   
        pool = Pool(processes=1) 
        pool.map(run_dataset, MFVIs[6:7])  