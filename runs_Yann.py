import os
from multiprocessing import Pool

if __name__ == "__main__":

    def run_script(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
    pool = Pool(processes=1) 


    pool.map(run_script, ["runs_Yann-Fun-splits.py"])  

    pool.map(run_script, ["runs_Yann-GeN-splits.py"])  
    
    

