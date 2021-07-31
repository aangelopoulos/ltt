import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
import torchvision as tv
import argparse
import time
import numpy as np
from scipy.stats import binom
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import seaborn as sns
from utils import *
from core.bounds import hb_p_value
from core.concentration import *
from statsmodels.stats.multitest import multipletests
import pdb

# Table will be n x m x N x N, where n is number of samples, m is number of losses, and N is sampling of lambda
def get_loss_tables(data,lambda1s,lambda2s):
    os.makedirs('./.cache/', exist_ok=True)
    try:
        loss_tables = torch.load('./.cache/loss_tables.pt')
        size_table = torch.load('./.cache/size_table.pt')
        frac_ind_ood_table = torch.load('./.cache/frac_ind_ood_table.pt')
        frac_ood_ood_table = torch.load('./.cache/frac_ood_ood_table.pt')
    except FileNotFoundError:
        # Load data
        odin_ind = data['odin_ind']
        odin_ind, ind_sort = odin_ind.sort()
        odin_ood = data['odin_ood']
        odin_ood, ood_sort = odin_ood.sort()
        softmax_ind = data['softmax_ind'][ind_sort]
        softmax_ood = data['softmax_ood'][ood_sort]
        labels_ind = data['labels_ind'][ind_sort]
        labels_ood = data['labels_ood'][ood_sort]
        # Preallocate space
        loss_tables = torch.zeros((softmax_ind.shape[0],2,lambda1s.shape[0],lambda2s.shape[0]))
        size_table = torch.zeros((softmax_ind.shape[0],lambda1s.shape[0],lambda2s.shape[0]))
        frac_ind_ood_table = torch.zeros((lambda1s.shape[0],))
        frac_ood_ood_table = torch.zeros((lambda1s.shape[0],))
        print("Calculating loss tables.")
        for i in tqdm(range(lambda1s.shape[0])):
            for j in range(lambda2s.shape[0]):
                num_incorrect_ind = (odin_ind > lambda1s[i]).float().sum()
                num_incorrect_ood = (odin_ood <= lambda1s[i]).float().sum()
                frac_ind_ood_table[i] = num_incorrect_ind/float(odin_ind.shape[0])
                frac_ood_ood_table[i] = 1-num_incorrect_ood/float(odin_ind.shape[0])
                _softmax_ind = softmax_ind[:-int(num_incorrect_ind)] 
                srtd, pi = _softmax_ind.sort(dim=1,descending=True)
                sizes = (srtd.cumsum(dim=1) <= lambda2s[j]).int().sum(dim=1)
                missed = ( sizes <= labels_ind[:-int(num_incorrect_ind)] ).int()
                loss_tables[:,0,i,j] = (odin_ind > lambda1s[i]).int()
                loss_tables[:-int(num_incorrect_ind),1,i,j] = missed 
                size_table[:-int(num_incorrect_ind),i,j] = sizes
                print(f"\n\rFrac InD OOD: {frac_ind_ood_table[i]}, Frac OOD OOD: {frac_ood_ood_table[i]}\033[1A",end="")
        torch.save(loss_tables,"./.cache/loss_tables.pt")
        torch.save(size_table,"./.cache/size_table.pt")
        torch.save(frac_ind_ood_table,"./.cache/frac_ind_ood_table.pt")
        torch.save(frac_ood_ood_table,"./.cache/frac_ood_ood_table.pt")

    return loss_tables, size_table, frac_ind_ood_table, frac_ood_ood_table

def trial_precomputed(loss_tables, alphas, delta, lambda1s, lambda2s, num_calib, maxiter):
    n = loss_tables.shape[0]
    perm = torch.randperm(n)
    
    loss_tables = loss_tables[perm]
    calib_tables, val_tables = (loss_tables[:num_calib], loss_tables[num_calib:])
    
    # Get p-values for each loss
    r_hats_risk1 = calib_tables[:,0,:].flatten(start_dim=1).mean(axis=0).squeeze() # empirical risk at each lambda combination
    p_values_risk1 = np.array([hb_p_value(r_hat,n,alphas[0]) for r_hat in r_hats_risk1])
    r_hats_risk2 = calib_tables[:,1,:].flatten(start_dim=1).mean(axis=0).squeeze() - alphas[1]*(1-r_hats_risk1) + alphas[1] # empirical risk at each lambda combination using trick
    p_values_risk2 = np.array([hb_p_value(r_hat,n,alphas[1]) for r_hat in r_hats_risk2])

    # Populate the corrected p-values
    p_values_corrected = np.zeros_like(p_values_risk1)
    for i in tqdm(range(p_values_risk1.shape[0])):
        if p_values_risk1[i] < delta and p_values_risk2[i] < delta:
            _, pvc, _, _ = multipletests(np.array([p_values_risk1[i], p_values_risk2[i]]), method='holm')
            p_values_corrected[i] = pvc.max()
            print(f"\nCorrected p-value: {pvc.max()}\033[1A\r",end="")
        else:
            p_values_corrected[i] = min(max(2*p_values_risk1[i], 2*p_values_risk2[i]),1)

    # Bonferroni correct over lambda to get the valid discoveries
    R = bonferroni(p_values_corrected, delta)

    # TODO: INDEX PROPERLY TO GET THE CORRECT LAMBDAS
    if R.shape[0] == 0:
        return 0.0, 0.0, 1.0

    lhat = lambdas[R.min()]

    val_predictions = val_scores > lhat

    pfdp = 1-val_corrects[val_predictions].float().mean()
    pfdp = np.nan_to_num(pfdp)
    
    mean_size = val_predictions.float().mean()
    
    return pfdp, mean_size, lhat

def experiment(alphas,delta,lambda1s,lambda2s,num_calib,num_trials,maxiter,cache_dir):
    df_list = []
    rejection_region_functions = (bonferroni_HB,multiscale_bonferroni_HB,romano_wolf_multiplier_bootstrap) 
    rejection_region_names = ("HBBonferroni","Multiscale HBBonferroni","RWMB")

    for idx in range(len(rejection_region_functions)):
        rejection_region_function = rejection_region_functions[idx]
        rejection_region_name = rejection_region_names[idx]
        fname = f'./.cache/{alphas}_{delta}_{num_calib}_{num_trials}_{rejection_region_name}_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","coverage","OOD Type I","OOD Type II","mean size","alpha1","alpha2","delta","region name"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            data = {}
            data['softmax_ind'] = torch.load(cache_dir + "softmax_scores_in_distribution.pt")
            data['softmax_ood'] = torch.load(cache_dir + "softmax_scores_out_of_distribution.pt")
            data['odin_ind'] = 1-torch.load(cache_dir + "ood_scores_in_distribution.pt")
            data['odin_ood'] = 1-torch.load(cache_dir + "ood_scores_out_of_distribution.pt")
            data['labels_ind'] = torch.load(cache_dir + "labels_in_distribution.pt")
            data['labels_ood'] = torch.load(cache_dir + "labels_out_of_distribution.pt")
            print('Dataset loaded')

            if lambda1s == None:
                lambda1s = torch.linspace(data['odin_ind'].min(),data['odin_ind'].max(),100)

            loss_tables, size_table, frac_ind_ood_table, frac_ood_ood_table = get_loss_tables(data,lambda1s,lambda2s)

            with torch.no_grad():
                local_df_list = []
                for i in tqdm(range(num_trials)):
                    pfdp, mean_size, lhat = trial_precomputed(loss_tables, alphas, delta, lambda1s, lambda2s, num_calib, maxiter)
                    dict_local = {"$\\hat{\\lambda}$": lhat,
                                    "coverage": pfdp,
                                    "OOD Type I": pfdp,
                                    "OOD Type II": pfdp,
                                    "mean size": mean_size,
                                    "alpha1": alphas[0],
                                    "alpha2": alphas[1],
                                    "delta": delta,
                                    "index": [0],
                                    "region name": rejection_region_name,
                                 }
                    df_local = pd.DataFrame(dict_local)
                    local_df_list = local_df_list + [df_local]
                df = pd.concat(local_df_list, axis=0, ignore_index=True)
                df.to_pickle(fname)

        df_list = df_list + [df]

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    cache_dir = './odin/code/.cache/' #TODO: Replace this with YOUR location of imagenet val set.

    alphas = [0.05,0.1]
    delta = 0.1
    maxiter = int(1e3)
    num_trials = 100 
    num_calib = 8000
    lambda1s = None 
    lambda2s = np.linspace(0,1,1000)
    
    experiment(alphas,delta,lambda1s,lambda2s,num_calib,num_trials,maxiter,cache_dir)
