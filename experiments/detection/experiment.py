import os, sys, inspect
sys.path.insert(1, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, gc, time, json, cv2, random, sys, traceback
from experiments.detection.utils import *
from core.bounds import hb_p_value
from core.concentration import *

import multiprocessing as mp

import pickle as pkl
from tqdm import tqdm
import pdb

import seaborn as sns

def plot(df_list,alphas,methods):
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(12,3))

    recalls = []
    mious = []
    mcvgs = []
    labels = []
    for i in range(len(df_list)):
        method = methods[i]
        if method == "Split Fixed Sequence":
            method = "Split Fixed\nSequence"
        df = df_list[i]
        recalls = recalls + [df['recall'],]
        mious = mious + [df['mIOU'],]
        mcvgs = mcvgs + [df['mean coverage'],]
        labels = labels + [method,]
        violations = (df['mean coverage'] < (1-alphas[0])) | (df['mIOU'] < (1-alphas[1])) | (df['recall']< (1-alphas[2]))
        print(f'{method}: fraction of violations is {violations.mean()}')

    sns.violinplot(data=recalls,ax=axs[0],orient='h',inner=None)
    sns.violinplot(data=mious,ax=axs[1],orient='h',inner=None)
    sns.violinplot(data=mcvgs,ax=axs[2],orient='h',inner=None)

    # Limits, lines, and labels
    axs[2].set_xlabel('Mean Coverage')
    axs[2].axvline(x=1-alphas[0],c='#999999',linestyle='--',alpha=0.7)
    axs[2].locator_params(axis='x',nbins=4)
    axs[2].locator_params(axis='y',nbins=4)
    axs[1].set_xlabel('Mean IOU')
    axs[1].axvline(x=1-alphas[1],c='#999999',linestyle='--',alpha=0.7)
    axs[1].locator_params(axis='x',nbins=4)
    axs[1].locator_params(axis='y',nbins=4)
    axs[0].set_xlabel('Recall')
    axs[0].axvline(x=1-alphas[2],c='#999999',linestyle='--',alpha=0.7)
    axs[0].locator_params(axis='x',nbins=4)
    axs[0].locator_params(axis='y',nbins=4)
    axs[0].set_yticklabels(labels)
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    sns.despine(ax=axs[2],top=True,right=True)
    fig.tight_layout()
    os.makedirs("./outputs/histograms/",exist_ok=True)
    plt.savefig("./" + f"outputs/histograms/detector_{alphas[0]}_{alphas[1]}_delta_histograms".replace(".","_") + ".pdf")

def calculate_all_p_values(calib_tables, alphas):
    n = calib_tables.shape[0]
    # Get p-values for each loss
    r_hats = calib_tables.mean(axis=0).squeeze().flatten(start_dim=1) # empirical risk at each lambda combination
    p_values = np.zeros_like(r_hats)
    for r in range(p_values.shape[0]):
        for i in range(p_values.shape[1]): 
            p_values[r,i] = hb_p_value(r_hats[r,i],n,alphas[r])

    return p_values

def calculate_corrected_p_values(calib_tables, alphas):
    # Combine them
    p_values = calculate_all_p_values(calib_tables, alphas)
    p_values_corrected = p_values.max(axis=0)
    return p_values_corrected
    
def flatten_lambda_meshgrid(lambda1s,lambda2s,lambda3s):
    l1_meshgrid, l2_meshgrid, l3_meshgrid = torch.meshgrid((torch.tensor(lambda1s),torch.tensor(lambda2s), torch.tensor(lambda3s)))
    l1_meshgrid = l1_meshgrid.flatten()
    l2_meshgrid = l2_meshgrid.flatten()
    l3_meshgrid = l3_meshgrid.flatten()
    return l1_meshgrid, l2_meshgrid, l3_meshgrid

def split_fixed_sequence(calib_tables, alphas, delta):
    # Split the data
    n_calib = calib_tables.shape[0]
    n_coarse = n_calib//2
    perm = torch.randperm(n_calib)
    calib_tables = calib_tables[perm]
    coarse_tables, fine_tables = (calib_tables[:n_coarse],calib_tables[n_coarse:])
    p_values_coarse = calculate_all_p_values(coarse_tables, alphas)
    # Find a lambda for each value of beta that controls the risk best.
    num_betas = 200 
    betas = np.logspace(-9,0,num_betas)
    lambda_sequence = np.zeros_like(betas)
    for b in range(num_betas):
        beta = betas[b]
        differences = np.abs(p_values_coarse - beta)
        infty_norms = np.linalg.norm(differences, ord=np.inf, axis=0)
        lambda_sequence[b] = infty_norms.argmin()

    _, idx = np.unique(lambda_sequence, return_index=True)
    lambda_sequence_ordered = lambda_sequence[np.sort(idx)]

    # Now test these lambdas 
    fine_tables = fine_tables.flatten(start_dim=2)[:,:,lambda_sequence_ordered]
    p_values_fine = calculate_corrected_p_values(fine_tables, alphas)
    rejections = lambda_sequence_ordered[np.nonzero(p_values_fine < delta)[0]].astype(int)

    return rejections 

def trial(i, method, alphas, delta, lambda1s, lambda2s, lambda3s, l1_meshgrid, l2_meshgrid, l3_meshgrid, num_calib, loss_tables, risks, lhats):
    fix_randomness(seed=(i*10000))
    n = loss_tables["tensor"].shape[0]
    perm = torch.randperm(n)

    local_tables = loss_tables["tensor"][perm]
    calib_tables, val_tables = (local_tables[:num_calib], local_tables[num_calib:])

    if method == "Bonferroni":
        p_values_corrected = calculate_corrected_p_values(calib_tables, alphas)
        R = bonferroni(p_values_corrected, delta)

    elif method == "Split Fixed Sequence":
        R = split_fixed_sequence(calib_tables, alphas, delta)

    if R.shape[0] == 0:
        lhats[i] = np.array([1.0,1.0,1.0])
        risks[i] = np.array([0.0,0.0,0.0])
        loss_tables["curr_proc"] -= 1

    # Index the lambdas
    l1s = l1_meshgrid[R]
    l2s = l2_meshgrid[R]
    l3s = l3_meshgrid[R]

    l3 = l3s[l3s > l1s].min()
    l2 = l2s[(l3s > l1s) & (l3s==l3)].median()
    l1 = l1s[(l3s > l1s) & (l2s==l2) & (l3s==l3)].min()

    lhats[i] = np.array([l1,l2,l3])

    # Validate

    idx1 = torch.nonzero(np.abs(lambda1s-lhats[i][0]) < 1e-10)[0][0].item()
    idx2 = torch.nonzero(np.abs(lambda2s-lhats[i][1]) < 1e-10)[0][0].item()
    idx3 = torch.nonzero(np.abs(lambda3s-lhats[i][2]) < 1e-10)[0][0].item()

    risks[i] = val_tables[:,:,idx1,idx2,idx3].mean(dim=0)
    loss_tables["curr_proc"] -= 1
    del calib_tables
    del val_tables
    del local_tables
    gc.collect()

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    num_trials = 1000 
    num_calib = 3000 
    num_processes = 30 
    mp.set_start_method('fork')
    alphas = [0.25, 0.5, 0.5] # neg_m_coverage, neg_miou, neg_recall
    delta = 0.1
    lambda1s = torch.linspace(0.5,0.8,50) # Top score threshold
    lambda2s = torch.linspace(0.3,0.7,5) # Segmentation threshold
    lambda3s = torch.logspace(-0.00436,0,25) # APS threshold

    # Multiprocessing setup
    manager = mp.Manager()
    loss_tables = manager.dict({"tensor": None, "curr_proc": 0})

    df_list = []
    methods = ["Bonferroni", "Split Fixed Sequence"]
    for method in methods:
        fname = f'./.cache/{method}_{alphas}_{delta}_{num_calib}_{num_trials}_dataframe.pkl'
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            with torch.no_grad():
                # Load cache
                with open('./.cache/loss_tables.pt', 'rb') as f:
                    #loss_tables["tensor"] = torch.tensor(np.random.random(size=(num_calib*2,3,lambda1s.shape[0],lambda2s.shape[0],lambda3s.shape[0])))/10
                    loss_tables["tensor"] = torch.load(f)

                risks = manager.dict({k:np.zeros((3,)) for k in range(num_trials)})
                lhats = manager.dict({k:np.zeros((3,)) for k in range(num_trials)})
                l1_meshgrid, l2_meshgrid, l3_meshgrid = flatten_lambda_meshgrid(lambda1s,lambda2s,lambda3s)

                # Test trial
                #trial(0, method, alphas, delta, lambda1s, lambda2s, lambda3s, l1_meshgrid, l2_meshgrid, l3_meshgrid, num_calib, loss_tables, risks, lhats)
                # Queue the jobs
                jobs = []
                for i in range(num_trials):
                    p = mp.Process(target = trial, args = (i, method, alphas, delta, lambda1s, lambda2s, lambda3s, l1_meshgrid, l2_meshgrid, l3_meshgrid, num_calib, loss_tables, risks, lhats)) 
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
                                  "mean coverage": 1-risks[i][0].item(),
                                  "mIOU": 1-risks[i][1].item(),
                                  "recall": 1-risks[i][2].item(),
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
        df_list = df_list + [df,] 
        average_lambda = np.concatenate([arr[None,:] for arr in df["$\\hat{\\lambda}$"].tolist()],axis=0).mean(axis=0)
        print(f"{method}: the average lambda_hat from the runs was: {list(average_lambda)}!")
    plot(df_list,alphas,methods)
    print("Done!")
