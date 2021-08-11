import os
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import pdb
import time

def mu(shm_name, mat_shape, mat_dtype, i,return_dict):
    print(f"Started mu process {i}!")
    existing_shm = shared_memory.SharedMemory(shm_name)
    mat = np.ndarray(mat_shape,dtype=mat_dtype,buffer=existing_shm.buf)
    my_mu = mat.mean()
    return_dict[i] = my_mu
    print(f"Mean at process {i}: {my_mu}")
    existing_shm.close()

if __name__ == '__main__':
    mp.set_start_method('fork') 
    mat = np.random.rand(1000,1000,1000)
    shm = shared_memory.SharedMemory(create=True, size=mat.nbytes)
    buf = np.ndarray(mat.shape, dtype=mat.dtype, buffer=shm.buf)
    buf[:] = mat[:]
    num_test = 1000
    manager = mp.Manager()
    return_dict = manager.dict({ k:0 for k in range(num_test)})
    jobs = []
    t1 = time.time()
    for i in range(num_test):
        p = mp.Process(target=mu, args=(shm.name, mat.shape, mat.dtype, i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    t2 = time.time()

    # NON-PARALLEL IMPLEMENTATION
    means = np.zeros((num_test,))
    for i in range(num_test):
        means[i] = mat.mean()

    t3 = time.time()
    print(f"Parallel version took: {t2-t1}. Single thread took: {t3-t2}")

    shm.close()
    shm.unlink()
