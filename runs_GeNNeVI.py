import os
from multiprocessing import Pool

if __name__ == "__main__":

    
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
    pool = Pool(processes=1) 
    
    for dataset in ['foong', 'foong_mixed', 'foong_sparse']:
        pool.map(run_dataset, ["-m Experiments.GeNNeVI-mr --nb_models=5 --setup="+dataset+" --n_samples_KL=500 --device='cuda:0'"])
        print(dataset+': done :-)')

    
    for dataset in ['wine','boston', 'energy','yacht', 'concrete','powerplant', 'kin8nm']:
        pool.map(run_dataset, ["-m Experiments.GeNNeVI-sp --setup="+dataset+" --n_samples_KL=500 --device='cuda:0'"])  
        print(dataset+': done :-)')

