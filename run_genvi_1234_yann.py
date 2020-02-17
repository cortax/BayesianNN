import os
from multiprocessing import Pool

if __name__ == "__main__":

   
    # (max_iter, learning_rate, patience, lat_dim, layerwidth, gpu)
    GeNVIs_params = [(20000,0.05,500,1, 50, 0), (20000,0.05,500,2,50, 0), (20000,0.05,500,3,50, 0), (20000,0.05,500,4,50, 0), (20000,0.05,500,5,50, 0)]

    
    GeNVIs = ["-m Experiments.GeNVI --max_iter=" + str(i) + " --learning_rate=" + str(j) + " --patience=" + str(k) + " --verbose=1 --device=cuda:" + str(q) + " --setup=foong --lat_dim=" + str(l) + " --layerwidth=" + str(p) for i,j,k,l,p,q in GeNVIs_params]
    
    
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
    
    pool = Pool(processes=1) 
    pool.map(run_dataset, GeNVIs[0:1])  
    pool = Pool(processes=1) 
    pool.map(run_dataset, GeNVIs[1:2])   
    pool = Pool(processes=1) 
    pool.map(run_dataset, GeNVIs[2:3])   
    pool = Pool(processes=1) 
    pool.map(run_dataset, GeNVIs[3:4])   
    pool = Pool(processes=1) 
    pool.map(run_dataset, GeNVIs[4:5])   