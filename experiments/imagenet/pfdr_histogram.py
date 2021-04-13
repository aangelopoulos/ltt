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
from utils import *
import seaborn as sns
from core.concentration import *
import pdb

def trial_precomputed(top_scores, corrects, alpha, delta, num_lam, num_calib, maxiter):
    total=top_scores.shape[0]
    m=1000
    perm = torch.randperm(total)
    top_scores = top_scores[perm]
    corrects = corrects[perm].float()
    calib_scores, val_scores = (top_scores[0:num_calib], top_scores[num_calib:])
    calib_corrects, val_corrects = (corrects[0:num_calib], corrects[num_calib:])

    calib_scores, indexes = calib_scores.sort()
    calib_corrects = calib_corrects[indexes] 
    calib_accuracy = (calib_corrects.flip(dims=(0,)).cumsum(dim=0)/(torch.tensor(range(num_calib))+1)).flip(dims=(0,))
    calib_abstention_freq = (torch.tensor(range(num_calib))+1).float().flip(dims=(0,))/num_calib
    pfdp_pluses = torch.tensor( [ pfdr_ucb(num_calib, m, calib_accuracy[i], calib_abstention_freq[i], delta, maxiter) for i in tqdm(range(300)) ] )

    pfdr = 0.2 
    sizes_mean = 0.1 
    lhat = 0.7 

    return pfdr, sizes_mean, lhat

def experiment(alpha,delta,num_lam,num_calib,num_trials,maxiter,imagenet_val_dir):
    df_list = []
    fname = f'.cache/{alpha}_{delta}_{num_lam}_{num_calib}_{num_trials}_dataframe.pkl'

    df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","pfdr","mean size","gamma","delta"])
    try:
        df = pd.read_pickle(fname)
    except FileNotFoundError:
        dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', imagenet_val_dir)
        print('Dataset loaded')
        
        classes_array = get_imagenet_classes()
        T = platt_logits(dataset_precomputed)
        
        logits, labels = dataset_precomputed.tensors
        top_scores, top_classes = (logits/T.cpu()).softmax(dim=1).max(dim=1)
        corrects = top_classes==labels

        with torch.no_grad():
            local_df_list = []
            for i in tqdm(range(num_trials)):
                pfdr, mean_size, lhat = trial_precomputed(top_scores, corrects, alpha, delta, num_lam, num_calib, maxiter)
                dict_local = {"$\\hat{\\lambda}$": lhat,
                                "pfdr": pfdr,
                                "mean size": mean_size,
                                "alpha": alpha,
                                "delta": delta,
                                "index": [0]
                             }
                df_local = pd.DataFrame(dict_local)
                local_df_list = local_df_list + [df_local]
            df = pd.concat(local_df_list, axis=0, ignore_index=True)
            df.to_pickle(fname)

    df_list = df_list + [df]

def platt_logits(calib_dataset, max_iters=10, lr=0.01, epsilon=0.01):
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=1024, shuffle=False, pin_memory=True) 
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    imagenet_val_dir = '/scratch/group/ilsvrc/val' #TODO: Replace this with YOUR location of imagenet val set.

    alphas = [0.1,0.05]
    deltas = [0.1,0.1]
    params = list(zip(alphas,deltas))
    num_lam = 100 
    maxiter = int(1e3)
    num_trials = 10
    num_calib = 30000
    
    for alpha, delta in params:
        print(f"\n\n\n ============           NEW EXPERIMENT alpha={alpha} delta={delta}           ============ \n\n\n") 
        experiment(alpha,delta,num_lam,num_calib,num_trials,maxiter,imagenet_val_dir)
