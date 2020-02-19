import os
from multiprocessing import Pool

if __name__ == "__main__":

   
    # (datasets, GPU)
    datasets = [("foong", "0"), ("boston", "1"), ("wine", "0"), ("concrete", "1"), ("kin8nm", "0"), ("yacht", "1"), ("powerplant", "0")]

    GeNVIs = ["-m Experiments.GeNVI --max_iter=40000 --n_samples_ED=75 --n_samples_LP=100 --lat_dim=5 --layerwidth=50 --learning_rate=0.06 --patience=800 --lr_decay=.6 --verbose=1 --device=cuda:" + j + " --setup=" + i for i,j in datasets]
    
    
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
    
    for i in range(0,10):
        pool = Pool(processes=2) 
        pool.map(run_dataset, GeNVIs[0:2])  
        pool = Pool(processes=2) 
        pool.map(run_dataset, GeNVIs[2:4])   
        pool = Pool(processes=2) 
        pool.map(run_dataset, GeNVIs[4:6])   
        pool = Pool(processes=1) 
        pool.map(run_dataset, GeNVIs[6:7])