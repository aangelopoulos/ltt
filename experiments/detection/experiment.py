# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, time, json, cv2, random, sys, traceback
from utils import *
from core.bounds import hb_p_value

import multiprocessing as mp

import pickle as pkl
from tqdm import tqdm
import pdb

import seaborn as sns

def calculate_corrected_p_values(calib_tables, alphas):
    n = calib_tables.shape[0]
    # Get p-values for each loss
    r_hats = calib_tables.mean(axis=0).squeeze().flatten(start_dim=1) # empirical risk at each lambda combination
    p_values = np.array_like(r_hats)
    for i in range(p_values.shape[1]):
        for j in range(p_values.shape[2]):
            for k in range(p_values.shape[3]):
                for r in range(p_vales.shape[0]): 
                    p_values[r,i,j,k] = hb_p_value(r_hats[r,i,j,k],n,alphas[r])

    # Combine them
    p_values_corrected = p_values.max(dim=0)
    return p_values_corrected
    
def flatten_lambda_meshgrid(lambda1s,lambda2s,lambda3s):
    l1_meshgrid, l2_meshgrid, l3_meshgrid = torch.meshgrid(torch.tensor(lambda1s),torch.tensor(lambda2s), torch.tensor(lambda3s))
    l1_meshgrid = l1_meshgrid.flatten()
    l2_meshgrid = l2_meshgrid.flatten()
    l3_meshgrid = l3_meshgrid.flatten()
    return l1_meshgrid, l2_meshgrid, l3_meshgrid


def trial(i, alphas, delta, lambda1s, lambda2s, lambda3s, num_calib, loss_tables, risks, lhats):
    fix_randomness(seed=(i*num_calib))
    n = loss_tables["tensor"].shape[0]
    perm = torch.randperm(n)

    local_tables = loss_tables["tensor"][perm]
    calib_tables, val_tables = (local_tables[:num_calib], local_tables[num_calib:])
    l1_meshgrid, l2_meshgrid, l3_meshgrod = flatten_lambda_meshgrid(lambda1s,lambda2s,lambda3s)

    p_values_corrected = calculate_corrected_p_values(calib_tables, alphas, lambda1s, lambda2s, lambda3s)
    R = bonferroni(p_values_corrected, delta)

    if R.shape[0] == 0:
        return 0.0, 0.0, 0.0, np.array([1.0,1.0,1.0])

    # Index the lambdas
    l1s = l1_meshgrid[R]
    l2s = l2_meshgrid[R]
    l3s = l3_meshgrid[R]
    
    l1 = l1s.min()
    l2 = l2s[l1s==l1].min()
    l3 = l3s[(l1s==l1) & (l2s=l2)].min()
    lhats[i] = np.array([l1,l2,l3])

    # Validate
    idx1 = np.nonzero(np.abs(lambda1s-lhats[i][0]) < 1e-10)[0]
    idx2 = np.nonzero(np.abs(lambda2s-lhats[i][1]) < 1e-10)[0]
    idx3 = np.nonzero(np.abs(lambda3s-lhats[i][2]) < 1e-10)[0]

    risks[i] = 1-val_tables[:,0,idx1,idx2,idx3].mean()
    loss_tables["curr_proc"] -= 1

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    num_trials = 1000
    num_calib = 3000
    num_processes = 30
    mp.set_start_method('fork')
    alphas = [0.2, 0.5, 0.5] # neg_m_coverage, neg_miou, neg_recall
    delta = 0.1
    lambda1s = torch.linspace(0.5,1,5) # Top score threshold
    lambda2s = torch.linspace(0,1,10) # Segmentation threshold
    lambda3s = torch.tensor([0.9,0.925,0.95,0.975,0.99,0.995,0.999,0.9995,1]) # APS threshold

    # Multiprocessing setup
    manager = mp.Manager()
    loss_tables = manager.dict({"tensor": None, "curr_proc": 0})

    fname = f'./.cache/{alphas}_{delta}_{num_calib}_{num_trials}_dataframe.pkl'

    try:
        df = pd.read_pickle(fname)
    except FileNotFoundError:
        with torch.no_grad():
            # Load cache
            with open('./.cache/loss_tables.pt', 'rb') as f:
                loss_tables["tensor"] = torch.load(f)

            risks = manager.dict({k:np.zeros((3,)) for k in range(num_trials)})
            lhats = manager.dict({k:np.zeros((3,)) for k in range(num_trials)})

            # Queue the jobs
            jobs = []
            for i in range(num_trials):
                p = mp.Process(target = trial, args = (i, alphas, delta, lambda1s, lambda2s, lambda3s, num_calib, loss_tables, risks, lhats)) 
                jobs.append(p)

            pbar = tqdm(total=num_trials)

            # Run the jobs
            for proc in jobs:
                while loss_tables["curr_proc"] >= num_processes:
                    time.sleep(2)
                proc.start()
                loss_tables["curr_proc"] += 1
                pbar.update(1)

            pbar.close()

            for proc in jobs:
                proc.join()

            # Form the large dataframe
            local_df_list = []
            for i in tqdm(range(num_trials)):
                dict_local = {"$\\hat{\\lambda}$": [lhats[i],],
                              "mean coverage": 1-risks[0],
                              "mIOU": 1-risks[1],
                              "recall": 1-risks[2],
                              "alpha1": alphas[0],
                              "alpha2": alphas[1],
                              "alpha3": alphas[2],
                              "delta": delta,
                              "index": [0],
                             }
                df_local = pd.DataFrame(dict_local)
                local_df_list = local_df_list + [df_local]
            df = pd.concat(local_df_list, axis=0, ignore_index=True)
            df.to_pickle(fname)
    pdb.set_trace()
    print("Done!")
