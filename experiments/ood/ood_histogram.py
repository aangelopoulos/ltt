import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import ctypes
import torch
import torchvision as tv
import argparse
import time
import numpy as np
import scipy.sparse as sparse 
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
import multiprocessing as mp
import time

data = {}
global_dict = {"loss_tables": None}

def plot_histograms(df_list,alphas,delta):
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(12,3))
    coverages = []
    oodt1s = []
    labels = []
    for df in df_list:
        region_name = df['region name'][0]
        if region_name == "2D Fixed Sequence":
            region_name = '2D Fixed\nSequence'
        coverages = coverages + [df['coverage'],]
        oodt1s = oodt1s + [df['OOD Type I'],]
        labels = labels + [region_name,]
        axs[2].scatter(1-df['coverage'],df['OOD Type I'], alpha=0.7, label=region_name)
        fraction_violated = ((df['coverage'] < 1-alphas[1]) | (df['OOD Type I'] > alphas[0])).astype(float).mean()
        print(f"Fraction violated (at least one risk) using {region_name}: {fraction_violated}")
    sns.violinplot(data=coverages,ax=axs[0],orient='h',inner=None)
    sns.violinplot(data=oodt1s,ax=axs[1],orient='h',inner=None)

    # Limits, lines, and labels
    #axs[0].set_ylabel("Histogram Density")
    axs[0].set_xlabel("Coverage")
    axs[0].axvline(x=1-alphas[1],c='#999999',linestyle='--',alpha=0.7)
    axs[0].locator_params(axis="x", nbins=4)
    axs[0].set_yticklabels(labels)
    axs[1].set_xlabel("CIFAR marked OOD")
    axs[1].axvline(x=alphas[0],c='#999999',linestyle='--',alpha=0.7)
    axs[1].locator_params(axis="x", nbins=4)
    axs[2].axvline(x=alphas[1],c='#999999', linestyle='--', alpha=0.7)
    axs[2].axhline(y=alphas[0],c='#999999', linestyle='--', alpha=0.7)
    axs[2].legend(loc='lower left')
    axs[2].set_xlim(left=0,right=1.05*max([(1-df['coverage']).max() for df in df_list]))
    axs[2].set_ylim(bottom=0,top=1.05*max([df['OOD Type I'].max() for df in df_list]))
    axs[2].set_xlabel("Miscoverage")
    axs[2].set_ylabel("CIFAR Marked OOD")
    axs[2].locator_params(axis="x", nbins=4)
    axs[2].locator_params(axis="y", nbins=4)
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    sns.despine(ax=axs[2],top=True,right=True)
    fig.tight_layout()
    os.makedirs("./outputs/histograms",exist_ok=True)
    plt.savefig("./" + f"outputs/histograms/ood_{alphas[0]}_{alphas[1]}_{delta}_histograms".replace(".","_") + ".pdf")

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
            num_incorrect_ind = (odin_ind > lambda1s[i]).float().sum()
            num_incorrect_ood = (odin_ood <= lambda1s[i]).float().sum()
            frac_ind_ood_table[i] = num_incorrect_ind/float(odin_ind.shape[0])
            frac_ood_ood_table[i] = 1-num_incorrect_ood/float(odin_ind.shape[0])
            if i > 0 and frac_ind_ood_table[i] == frac_ind_ood_table[i-1]:
                loss_tables[:,:,i,:] = loss_tables[:,:,i-1,:]
                size_table[:,i,:] = size_table[:,i-1,:]
            else:
                for j in range(lambda2s.shape[0]):
                    if num_incorrect_ind == 0:
                        index_split = None
                    else:
                        index_split = -int(num_incorrect_ind)
                    _softmax_ind = softmax_ind[:index_split] 
                    if _softmax_ind.shape[0] > 0:
                        srtd, pi = _softmax_ind.sort(dim=1,descending=True)
                        sizes = (srtd.cumsum(dim=1) <= lambda2s[j]).int().sum(dim=1)
                        sizes = torch.max(sizes,torch.ones_like(sizes))
                        rank_of_true = (pi == labels_ind[:index_split,None]).int().argmax(dim=1) + 1
                        missed = ( sizes < rank_of_true ).int()
                        loss_tables[:index_split,1,i,j] = missed 
                        size_table[:index_split,i,j] = sizes
                loss_tables[:,0,i,:] = (odin_ind > lambda1s[i]).int().unsqueeze(dim=1)
            print(f"\n\ri: {i}, Frac InD OOD: {frac_ind_ood_table[i]}, Frac OOD OOD: {frac_ood_ood_table[i]}\033[1A",end="")
        torch.save(loss_tables,"./.cache/loss_tables.pt")
        torch.save(size_table,"./.cache/size_table.pt")
        torch.save(frac_ind_ood_table,"./.cache/frac_ind_ood_table.pt")
        torch.save(frac_ood_ood_table,"./.cache/frac_ood_ood_table.pt")
        print("Loss tables calculated!")

    return loss_tables, size_table, frac_ind_ood_table, frac_ood_ood_table

def calculate_corrected_p_values(calib_tables, alphas, lambda1s, lambda2s):
    n = calib_tables.shape[0]
    # Get p-values for each loss
    r_hats_risk1 = calib_tables[:,0,:].mean(axis=0).squeeze().flatten() # empirical risk at each lambda combination
    p_values_risk1 = np.array([hb_p_value(r_hat,n,alphas[0]) for r_hat in r_hats_risk1])
    r_hats_risk2 = (calib_tables[:,1,:] * (1-calib_tables[:,0,:]) - alphas[1]*(1-calib_tables[:,0,:])).mean(axis=0).squeeze().flatten() + alphas[1] # empirical risk at each lambda combination using trick
    p_values_risk2 = np.array([hb_p_value(r_hat,n,alphas[1]) for r_hat in r_hats_risk2])

    # Combine them
    p_values_corrected = np.maximum(p_values_risk1,p_values_risk2) 
    return p_values_corrected
    
def flatten_lambda_meshgrid(lambda1s,lambda2s):
    l1_meshgrid, l2_meshgrid = torch.meshgrid(torch.tensor(lambda1s),torch.tensor(lambda2s))
    l1_meshgrid = l1_meshgrid.flatten()
    l2_meshgrid = l2_meshgrid.flatten()
    return l1_meshgrid, l2_meshgrid

def getA_gridsplit(lambda1s,lambda2s):
    l1_meshgrid, l2_meshgrid = torch.meshgrid(torch.tensor(lambda1s),torch.tensor(lambda2s))
    Is = torch.tensor(range(1,lambda1s.shape[0]+1)).flip(dims=(0,)).double()
    Js = torch.tensor(range(1,lambda2s.shape[0]+1)).flip(dims=(0,)).double()
    Is, Js = torch.meshgrid(Is,Js)
    #Wc[i,j]=mass from node [i,j] to node [i,j-1]
    Wc = torch.zeros_like(l1_meshgrid)
    Wc[:] = 1
    data = Wc.flatten().numpy()
    row = np.array(range(Wc.numel()))
    i_orig = row // Wc.shape[1]
    j_orig = row % Wc.shape[1]
    col = i_orig*Wc.shape[1] + j_orig - 1
    idx = (col >= 0) & (col < Wc.numel()) 
    data = data[idx]
    row = row[idx]
    col = col[idx]
    # Main edges
    A = sparse.csr_matrix((data, (row, col)), shape=(Wc.numel(), Wc.numel()))
    
    # Skip edges 
    #skip_bool = (np.array(range(A.shape[0])) % lambda2s.shape[0])==0
    #skip_bool2 = (np.array(range(A.shape[0])) % lambda2s.shape[0])==(lambda2s.shape[0]-1)
    #A[skip_bool,:] = 0
    #A[skip_bool,skip_bool2] = 1
    A.eliminate_zeros()

    # Set up the error budget
    error_budget = torch.zeros((lambda1s.shape[0],lambda2s.shape[0]))
    error_budget[:,-1] = delta/lambda1s.shape[0]
    return A, error_budget 

def getA_row_equalized(lambda1s, lambda2s):
    l1_meshgrid, l2_meshgrid = torch.meshgrid(torch.tensor(lambda1s),torch.tensor(lambda2s))
    Is = torch.tensor(range(1,lambda1s.shape[0]+1)).flip(dims=(0,)).double()
    Js = torch.tensor(range(1,lambda2s.shape[0]+1)).flip(dims=(0,)).double()
    Is, Js = torch.meshgrid(Is,Js)
    #Wr[i,j]=mass from node [i,j] to node [i-1,j]
    #Wc[i,j]=mass from node [i,j] to node [i,j-1]
    Wr = torch.zeros_like(l1_meshgrid)
    Wc = torch.zeros_like(l1_meshgrid)
    small_axis = min(lambda1s.shape[0],lambda2s.shape[0])
    large_axis = max(lambda1s.shape[0],lambda2s.shape[0])
    tri_bool = (Is + Js) <= small_axis
    Wr[tri_bool] = (Is/(Is+Js))[tri_bool]
    Wc[tri_bool] = (Js/(Is+Js))[tri_bool]
    Wc[~tri_bool & (Js < large_axis)] = 1
    Wr[Js == large_axis] = 1
    data = torch.cat((Wr.flatten(),Wc.flatten()),dim=0).numpy()
    row = np.concatenate((np.array(range(Wr.numel())),np.array(range(Wr.numel()))),axis=0)
    i_orig = row // Wr.shape[1]
    j_orig = row % Wr.shape[1]
    col = np.concatenate((
            (i_orig[:row.shape[0]//2] - 1)*Wr.shape[1] + j_orig[:row.shape[0]//2], (i_orig[row.shape[0]//2:])*Wr.shape[1] + j_orig[row.shape[0]//2:] - 1
        ), axis=0)
    idx = (col >= 0) & (col < Wr.numel()) 
    data = data[idx]
    row = row[idx]
    col = col[idx]
    A = sparse.csr_matrix((data, (row, col)), shape=(Wr.numel(), Wr.numel()))

    # Set up the error budget
    error_budget = torch.zeros((lambda1s.shape[0],lambda2s.shape[0]))
    error_budget[-1,-1] = delta
    return A, error_budget 

def to_flat_index(idxs,shape):
    return idxs[0]*shape[1] + idxs[1]

def to_rect_index(idxs,shape):
    return [idxs//shape[1], idxs % shape[1]]

def coordsplit_test(loss_tables, alphas, delta, lambda1s, lambda2s, num_calib):
    r_hats_risk1 = loss_tables[:,0,:].mean(dim=0)
    r_hats_risk2 = (loss_tables[:,1,:] * (1-loss_tables[:,0,:]) - alphas[1]*(1-loss_tables[:,0,:])).mean(dim=0) + alphas[1] # empirical risk at each lambda combination using trick
    r_hats = torch.cat((r_hats_risk1[None,:],r_hats_risk2[None,:]),dim=0)
    p_vals = torch.ones_like(r_hats)
    # Calculate the p-values
    for (r, i, j), r_hat in np.ndenumerate(r_hats):
        if r_hat > alphas[r]:
            continue # assign a p-value of 1 
        p_vals[r,i,j] = hb_p_value(r_hat,num_calib,alphas[r])

    lambda1_idx = np.argmax(p_vals[0,:,0] < delta/2).item()
    lambda2_idx = np.argmax(p_vals[1,lambda1_idx,:] < delta/2).item()
    R = np.array([lambda1_idx*lambda2s.shape[0] + lambda2_idx,])
    return R

def graph_test(A, error_budget, loss_tables, alphas, delta, lambda1s, lambda2s, num_calib, acyclic=False):
    r_hats_risk1 = loss_tables[:,0,:].mean(dim=0)
    r_hats_risk2 = (loss_tables[:,1,:] * (1-loss_tables[:,0,:]) - alphas[1]*(1-loss_tables[:,0,:])).mean(dim=0) + alphas[1] # empirical risk at each lambda combination using trick
    r_hats = torch.cat((r_hats_risk1[None,:],r_hats_risk2[None,:]),dim=0)
    p_vals = torch.ones_like(r_hats)
    # Calculate the p-values
    for (r, i, j), r_hat in np.ndenumerate(r_hats):
        if r_hat > alphas[r]:
            continue # assign a p-value of 1 
        p_vals[r,i,j] = hb_p_value(r_hat,num_calib,alphas[r])
    p_vals = p_vals.max(dim=0)[0]
    
    rejected_bool = torch.zeros_like(p_vals) > 1 # all false

    A = A.tolil()
    if not acyclic:
        A_csr = A.tocsr()
    # Graph updates
    while(rejected_bool.int().sum() < p_vals.numel() and error_budget.sum() > 0):
        argmin = (p_vals/error_budget).argmin()
        argmin_rect = to_rect_index(argmin.numpy(),p_vals.shape)
        minval = p_vals[argmin_rect[0],argmin_rect[1]] 
        #print(f"discoveries: {rejected_bool.float().sum():.3f}, error left total: {error_budget.sum():.3e}, point:{argmin_rect}, error here: {error_budget[argmin_rect[0],argmin_rect[1]]:.3e}, p_val: {minval:.3e}")
        if minval > error_budget[argmin_rect[0],argmin_rect[1]]:
            error_budget[argmin_rect[0],argmin_rect[1]] = 0
            continue
        rejected_bool[argmin_rect[0],argmin_rect[1]] = True
        # Modify the graph 
        outgoing_edges = A[argmin,:]
        for e in range(len(outgoing_edges.data[0])):
            g_jl = outgoing_edges.data[0][e]
            destination = to_rect_index(outgoing_edges.rows[0][e],error_budget.shape)
            error_budget[destination[0],destination[1]] += g_jl*error_budget[argmin_rect[0],argmin_rect[1]]
        if not acyclic:
            incoming_edges = A_csr[:,argmin].T # Use CSR here for speed
            nodes_to_update = list(set(outgoing_edges.rows[0] + list(incoming_edges.indices))) #Incoming edges
            for node in nodes_to_update:
                if node == argmin.item():
                    continue
                g_lj = incoming_edges[0,node]
                if g_lj == 0:
                    continue
                A[node,:] = (A[node,:] + g_lj*outgoing_edges)/(1-g_lj*outgoing_edges[0,node])

        A[:,argmin] = 0
        #A[argmin,:] = 0 # No incoming nodes, so don't have to set this.
        if not acyclic:
            A_csr = A.tocsr()
        error_budget[argmin_rect[0],argmin_rect[1]] = 0.0

    return rejected_bool.flatten().nonzero()

def gridsplit_graph_test(loss_tables, alphas, delta, lambda1s, lambda2s, num_calib):
    A, error_budget = getA_gridsplit(lambda1s,lambda2s)
    return graph_test(A, error_budget, loss_tables, alphas, delta, lambda1s, lambda2s, num_calib, acyclic=True)

def row_equalized_graph_test(loss_tables, alphas, delta, lambda1s, lambda2s, num_calib):
    A, error_budget = getA_row_equalized(lambda1s,lambda2s)
    return graph_test(A, error_budget, loss_tables, alphas, delta, lambda1s, lambda2s, num_calib, acyclic=True)

def trial_precomputed(method_name, alphas, delta, lambda1s, lambda2s, num_calib, maxiter, i, r1, r2, oodt2, lht, curr_proc_dict):
    fix_randomness(seed=(i*num_calib))
    n = global_dict['loss_tables'].shape[0]
    perm = torch.randperm(n)
    
    loss_tables = global_dict['loss_tables'][perm]
    calib_tables, val_tables = (loss_tables[:num_calib], loss_tables[num_calib:])
    l1_meshgrid, l2_meshgrid = flatten_lambda_meshgrid(lambda1s,lambda2s)
    lambda_selector = np.ones((lambda1s.shape[0]*lambda2s.shape[0],)) > 2  # All false

    if method_name == "Hamming":
        R = row_equalized_graph_test(loss_tables, alphas, delta, lambda1s, lambda2s, num_calib)    
        lambda_selector[:] = True

    elif method_name == "Gridsplit SGT":
        R = gridsplit_graph_test(loss_tables, alphas, delta, lambda1s, lambda2s, num_calib)    
        lambda_selector[:] = True

    elif method_name == "2D Fixed Sequence":
        R = coordsplit_test(loss_tables, alphas, delta, lambda1s, lambda2s, num_calib)

        lambda_selector[:] = True

    else:
        if method_name == "Multiscale HBBonferroni":
            n_coarse = int(calib_tables.shape[0]/10)
            coarse_tables, fine_tables = (calib_tables[:n_coarse], calib_tables[n_coarse:])
            p_values_coarse = calculate_corrected_p_values(coarse_tables, alphas, lambda1s, lambda2s)
            # Get a band around delta that contains about 5% of examples.
            delta_quantile = (p_values_coarse <= delta).mean()
            lambda_selector[p_values_coarse <= 1.5*delta] = True
            frac_selected = lambda_selector.astype(float).mean()
            if frac_selected == 0:
                print("Selection failed!")
                lambda_selector[:] = True 
            else:
                p_values_corrected = calculate_corrected_p_values(fine_tables, alphas, lambda1s, lambda2s)
        else:
            p_values_corrected = calculate_corrected_p_values(calib_tables, alphas, lambda1s, lambda2s)
            lambda_selector[:] = True

        if method_name == "Fallback":
            p_values_corrected = p_values_corrected.reshape((lambda1s.shape[0],lambda2s.shape[0]))
            mask = np.zeros_like(p_values_corrected)
            for row in range(p_values_corrected.shape[0]):
                p_value_exceed_indexes = np.nonzero(p_values_corrected[row,:] > (delta/lambda1s.shape[0]))[0]
                valid_col = min(p_value_exceed_indexes.max()+1,p_values_corrected.shape[1]-1)
                if valid_col == 999:
                    continue
                mask[row,valid_col] = 1
            R = np.nonzero(mask.flatten())[0]
            #R = np.nonzero(p_values_corrected < (delta / lambda1s.shape[0]))[0]
        else:
            # Bonferroni correct over lambda to get the valid discoveries
            R = bonferroni(p_values_corrected[lambda_selector], delta)

    if R.shape[0] == 0:
        return 0.0, 0.0, 0.0, np.array([1.0,1.0]) 

    # Index the lambdas
    l1_meshgrid = l1_meshgrid[lambda_selector]
    l2_meshgrid = l2_meshgrid[lambda_selector]
    l1s = l1_meshgrid[R]
    l2s = l2_meshgrid[R]

    minrow = (R//lambda2s.shape[0]).min()
    mincol = (R %lambda2s.shape[0]).min()
    print(minrow)
    print(mincol)

    lhat = np.array([l1s[l2s==l2s.min()].min(), l2s.min()])
    #lhat = np.array([l1s.min(), l2s[l1s==l1s.min()].min()])
    print(f"Region: {method_name}, Lhat: {lhat}")

    # Validate
    idx1 = np.nonzero(np.abs(lambda1s-lhat[0]) < 1e-10)[0]
    idx2 = np.nonzero(np.abs(lambda2s-lhat[1]) < 1e-10)[0] 

    num_ood = val_tables[:,0,idx1,idx2].sum()
    risk1 = float(num_ood) / float(val_tables.shape[0])
    selector = -int(num_ood) if num_ood != 0 else None
    risk2 = val_tables[:selector,1,idx1,idx2].mean().item()
    
    ood_type2 = 1-global_dict['frac_ood_ood_table'][idx1].item()
    
    r1[i] = risk1
    r2[i] = risk2
    oodt2[i] = ood_type2
    lht[i] = lhat
    curr_proc_dict['num'] -= 1

# Define the tables in the global scope

def experiment(alphas,delta,lambda1s,lambda2s,num_calib,num_trials,maxiter,cache_dir,num_processes):
    df_list = []
    rejection_region_names = ("Bonferroni","2D Fixed Sequence","Fallback","Hamming")

    for idx in range(len(rejection_region_names)):
        rejection_region_name = rejection_region_names[idx]
        fname = f'./.cache/{alphas}_{delta}_{num_calib}_{num_trials}_{rejection_region_name}_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","coverage","OOD Type I","OOD Type II","alpha1","alpha2","delta","region name"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            data['softmax_ind'] = torch.load(cache_dir + "softmax_scores_in_distribution.pt")
            data['softmax_ood'] = torch.load(cache_dir + "softmax_scores_out_of_distribution.pt")
            data['odin_ind'] = 1-torch.load(cache_dir + "ood_scores_in_distribution.pt")
            data['odin_ood'] = 1-torch.load(cache_dir + "ood_scores_out_of_distribution.pt")
            data['labels_ind'] = torch.load(cache_dir + "labels_in_distribution.pt")
            data['labels_ood'] = torch.load(cache_dir + "labels_out_of_distribution.pt")
            print('Dataset loaded')
            
            lambda1s = np.linspace(np.quantile(data['odin_ind'],0.5),np.quantile(data['odin_ind'],1-alphas[0]),100) 

            # Load data 
            global_dict['loss_tables'], global_dict['size_table'], global_dict['frac_ind_ood_table'], global_dict['frac_ood_ood_table'] = get_loss_tables(data,lambda1s,lambda2s)

            with torch.no_grad():
                # Setup shared memory for experiments
                manager = mp.Manager()
                return_risk1 = manager.dict({ k:0. for k in range(num_trials)})
                return_risk2 = manager.dict({ k:0. for k in range(num_trials)})
                return_ood_type2 = manager.dict({ k:0. for k in range(num_trials)})
                return_lhat = manager.dict({ k:np.array([]) for k in range(num_trials)})
                curr_proc_dict = manager.dict({'num': 0})

                # Multiprocessing: https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing and https://docs.python.org/3/library/multiprocessing.shared_memory.html

                jobs = []

                # TEST TRIAL
                #trial_precomputed("Graph", alphas, delta, lambda1s, lambda2s, num_calib, maxiter, 0, return_risk1, return_risk2, return_ood_type2, return_lhat, curr_proc_dict)
                #print("Test trial complete.")
                #trial_precomputed("HBBFSearch", alphas, delta, lambda1s, lambda2s, num_calib, maxiter, 0, return_risk1, return_risk2, return_ood_type2, return_lhat, curr_proc_dict)
                #trial_precomputed("Coordinate Split", alphas, delta, lambda1s, lambda2s, num_calib, maxiter, 0, return_risk1, return_risk2, return_ood_type2, return_lhat, curr_proc_dict)

                for i in range(num_trials):
                    p = mp.Process(target=trial_precomputed, args=(rejection_region_name, alphas, delta, lambda1s, lambda2s, num_calib, maxiter, i, return_risk1, return_risk2, return_ood_type2, return_lhat, curr_proc_dict))
                    jobs.append(p)

                pbar = tqdm(total=num_trials)

                for proc in jobs:
                    while curr_proc_dict['num'] >= num_processes:
                        time.sleep(2)
                    proc.start()
                    curr_proc_dict['num'] += 1
                    pbar.update(1)

                pbar.close()

                for proc in jobs:
                    proc.join()

                # Form the large dataframe
                local_df_list = []
                for i in tqdm(range(num_trials)):
                    dict_local = {"$\\hat{\\lambda}$": [return_lhat[i],],
                                    "coverage": 1-return_risk2[i],
                                    "OOD Type I": return_risk1[i],
                                    "OOD Type II": return_ood_type2[i],
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
    plot_histograms(df_list,alphas,delta)

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)
    mp.set_start_method('fork') 

    cache_dir = './odin/code/.cache/' 

    alphas = [0.05,0.01]
    delta = 0.1
    maxiter = int(1e3)
    num_trials = 100
    num_calib = 8000
    num_processes = 15
    lambda1s = None 
    lambda2s = np.linspace(0,1,1000)
    
    experiment(alphas,delta,lambda1s,lambda2s,num_calib,num_trials,maxiter,cache_dir,num_processes)
