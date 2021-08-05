import os
import multiprocessing as mp
import numpy as np
import pdb

def mu(i,return_dict):
    print(f"Started mu process {i}!")
    my_mu = mat.mean()
    return_dict[i] = my_mu

if __name__ == '__main__':
    mp.set_start_method('fork') 
    mat = np.random.rand(1000,1000,1000)
    manager = mp.Manager()
    num_test = 1000
    return_dict = manager.dict({ k:0 for k in range(num_test)})
    jobs = []
    for i in range(num_test):
        p = mp.Process(target=mu, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict.values())

