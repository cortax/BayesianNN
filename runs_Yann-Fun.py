import os
from multiprocessing import Pool

if __name__ == "__main__":

    n=5
    
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
    pool = Pool(processes=1) 

    dataset='powerplant'
    print(dataset)
    pool.map(run_dataset, ["-m Experiments.FuNNeVI-mres --batch=500 --patience=100 --n_samples_FU=150 --nb_models="+str(n)+" --setup="+dataset+"  --device='cuda:0'"])  
    print(dataset+': done :-)')
    
    for dataset in ['boston', 'yacht', 'concrete','energy', 'wine']:
        print(dataset)
        pool.map(run_dataset, ["-m Experiments.FuNNeVI-mres --batch=100  --patience=100 --n_samples_FU=150 --nb_models="+str(n)+" --setup="+dataset+"  --device='cuda:0'"])  
        print(dataset+': done :-)')
    


    for dataset in ['foong','foong_mixed']:
        print(dataset)
        pool.map(run_dataset, ["-m Experiments.FuNNeVI-mres --batch=60 --patience=100 --n_samples_FU=50 --nb_models="+str(n)+" --setup="+dataset+"  --device='cuda:0'"])  
        print(dataset+': done :-)')
        
    dataset='foong_sparse'
    
    print(dataset)
    pool.map(run_dataset, ["-m Experiments.FuNNeVI-mres --n_samples_FU=50 --nb_models="+str(n)+" --setup="+dataset+"  --device='cuda:0'"])  
    print(dataset+': done :-)')

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
