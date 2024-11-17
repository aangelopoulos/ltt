import os, sys, inspect
sys.path.insert(1, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
import pathlib
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests
import pdb
from pathlib import Path
import pickle as pkl
import pandas as pd
from core.utils import *
from core.bounds import hb_p_value, HB_mu_plus, HB_mu_minus
from core.uniform_concentration import nu_plus
CACHE = str(Path(__file__).parent.absolute()) + '/.cache/'

"""
    ROMANO WOLF MASTER ALGORITHM
"""
# there should be N inputs (one for each possible choice in the grid of lambdas)
# and the subset scoring function should output a real number for each subset of {1,...,N}
def romano_wolf(inputs,subset_scoring_function):
    N = inputs.shape[0]
    S = set(range(N))
    R = set() 
    while len(R) < N:
        dR = S.intersection(set(np.nonzero(inputs <= subset_scoring_function(S))[0]))
        if len(dR) == 0:
            return np.array(list(R))
        else:
            S.difference_update(dR)
            R.update(dR)
    return np.array(list(R))

"""
    RW SPECIALIZATIONS
"""
# loss_table is an n x N table containing the loss of each point at value lambda
# lambdas is the grid of lambdas
# alpha is the desired loss
# delta is the high probability bound
def romano_wolf_HB(loss_table,lambdas,alpha,delta):
    n = loss_table.shape[0]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    def subset_scoring_function(S):
        return delta/len(S)
    return romano_wolf(p_values,subset_scoring_function)

def romano_wolf_CLT(loss_table,lambdas,alpha,delta):
    t_values = np.sqrt(loss_table.shape[0])*(alpha - loss_table.mean(axis=0))/loss_table.std(axis=0) 
    p_values = 1-stats.norm.cdf(t_values)
    def subset_scoring_function(S):
        return delta/len(S)
    return romano_wolf(p_values,subset_scoring_function)

def romano_wolf_multiplier_bootstrap(loss_table,lambdas,alpha,delta,B=100):
    n = loss_table.shape[0]
    N = loss_table.shape[1]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    z_table = loss_table - loss_table.mean(axis=0)[np.newaxis,:] # Broadcast so Z_{i,j}=l_{i,j}-mean_i(l_{i,j})
    es = np.random.random(size=(n,B))
    cs = np.zeros((B,N))
    # Bootstrapping
    for b in range(B):
        cs[b] = np.mean(z_table * es[:,b:b+1],axis=0)
    def subset_scoring_function(S):
        idx = np.array(list(S))
        subset = cs[:,idx]
        maxes = np.max(subset,axis=1)
        return -np.quantile(maxes,1-delta,interpolation='higher') # Weird negative due to flipped sign in romano-wolf algorithm 
    return romano_wolf(-(alpha-r_hats),subset_scoring_function)

"""
    BONFERRONI MASTER ALGORITHM
"""
def bonferroni(p_values,delta):
    rejections, _, _, _ = multipletests(p_values,delta,method='holm',is_sorted=False,returnsorted=False)
    R = np.nonzero(rejections)[0]
    return R 

"""
    MULTISCALE MASTER ALGORITHM
"""

def multiscaleify(method, frac_data_coarse, loss_table, lambdas, alpha, delta, *argv):
    if frac_data_coarse == None:
        frac_data_coarse = 0.1
    num_coarse = int(np.ceil(loss_table.shape[0] * frac_data_coarse))
    num_fine = loss_table.shape[0] - num_coarse
    small_table, big_table = (loss_table[:num_coarse,:], loss_table[num_coarse:,:])    
    
    r_hats_coarse = small_table.mean(axis=0) 
    p_values_upper = np.array([hb_p_value(r_hat, num_coarse, alpha) for r_hat in r_hats_coarse])
    # TODO: Fix the second piece of the mask
    lambda_binary_mask = (r_hats_coarse <= alpha + 0.05).astype(float) * (r_hats_coarse > alpha - 0.05).astype(float)
    #lambda_binary_mask[-1] = 1.0 # Always include the last one.
    argmin = np.argmin(r_hats_coarse)
    lambda_binary_mask[argmin] = 1.0 # Always include the minimal element as a safety
    #lambda_binary_mask = (p_values_upper < delta).astype(float)
    #print(f"Fraction of grid to search: {lambda_binary_mask.sum()/lambda_binary_mask.shape[0]}")
    lambda_indexes_to_search = np.nonzero(lambda_binary_mask)[0]

    indexes_fine_grid = method(big_table[:,lambda_indexes_to_search],lambdas[lambda_indexes_to_search],alpha,delta, *argv)
    if len(indexes_fine_grid) == 0:
        print("ZERO SIZE\n\n\n")
        return np.array([0.5,])
    return_indexes = lambda_indexes_to_search[indexes_fine_grid]
    return return_indexes 

"""
    ORACLE METHOD
"""

def oracle_HB(loss_table,lambdas,alpha,delta):
    n = loss_table.shape[0]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    R = set(np.nonzero(p_values < delta)[0])
    return R

"""
    BONFERRONI SPECIALIZATIONS 
"""
def bonferroni_HB(loss_table,lambdas,alpha,delta):
    n = loss_table.shape[0]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    return bonferroni(p_values,delta)

def multiscale_bonferroni_HB(loss_table,lambdas,alpha,delta,frac_data_coarse=None):
    return multiscaleify(bonferroni_HB,frac_data_coarse,loss_table,lambdas,alpha,delta)

def bonferroni_CLT(loss_table,lambdas,alpha,delta):
    t_values = np.sqrt(loss_table.shape[0])*(alpha - loss_table.mean(axis=0))/loss_table.std(axis=0) 
    p_values = 1-stats.norm.cdf(t_values)
    return bonferroni(p_values,delta)

"""
    BONFERRONI SEARCH MASTER ALGORITHM
"""
def bonferroni_search(p_values,delta,downsample_factor):
    N = p_values.shape[0]
    N_coarse = max(int(p_values.shape[0]/downsample_factor),1)
    # Downsample, making sure to include the endpoints.
    coarse_indexes = set(range(0,N,downsample_factor))
    coarse_indexes.update({0,N-1})
    R = set() 
    for idx in coarse_indexes:
        _idx = idx
        while _idx >= 0 and _idx < N and p_values[_idx] < delta/N_coarse:
            R.update({_idx})
            _idx = _idx + 1 
    return np.array(list(R))

"""
    BONFERRONI SEARCH SPECIALIZATIONS 
"""
def bonferroni_search_HB(loss_table,lambdas,alpha,delta,downsample_factor):
    n = loss_table.shape[0]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    return bonferroni_search(p_values,delta,downsample_factor)

def multiscale_bonferroni_search_HB(loss_table,lambdas,alpha,delta,downsample_factor,frac_data_coarse=None):
    return multiscaleify(bonferroni_search_HB,frac_data_coarse,loss_table,lambdas,alpha,delta,loss_table.shape[1])

def bonferroni_search_CLT(loss_table,lambdas,alpha,delta,downsample_factor):
    t_values = np.sqrt(loss_table.shape[0])*(alpha - loss_table.mean(axis=0))/loss_table.std(axis=0) 
    p_values = 1-stats.norm.cdf(t_values)
    return bonferroni_search(p_values,delta,downsample_factor)

def graphical_bonferroni_search_HB(loss_table, lambdas_list, alphas, delta, frac_data_split=None):
    # Example loss table shape:  [9000, 2, 100, 1000]
    # Split the table
    n = loss_table.shape[0]
    n1 = np.floor(n * frac_data_split)
    perm = torch.randperm(n)
    selection_table, calib_table = (loss_table[perm][:n1],loss_table[perm][n1:])

    out_list = torch.meshgrid([torch.tensor(lams) for lams in lambdas_list])
    out_list = [lams.flatten() for lams in out_list] 
    selection_rhats = selection_table.flatten(start_dim=2).mean(dim=0) # rhat
    p_values_selection = np.zeros_like(selection_rhats)

"""
    NAIVE ALGORITHM
"""
# Just select the set of lambdas where the 1-delta quantile of the loss table is below alpha.
def naive_rejection_region(loss_table,lambdas,alpha,delta):
    try:
        return np.array([np.nonzero(loss_table.mean(axis=0) < alpha)[0][0],np.nonzero(loss_table.mean(axis=0)<alpha)[0][-1]])
    except:
        return np.array([])

"""
    UNIFORM REGION (only returns endpoints) 
"""
def uniform_region(loss_table,lambdas,alpha,delta,m):
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda (FDP)
    if r_hats.min() >= alpha:
        return np.array([])
    starting_index = (r_hats < alpha).nonzero()[0][0]
    ending_index = (r_hats < alpha).nonzero()[0][-1]
    R = np.array([])
    sig_figs = int(np.ceil(np.log10(lambdas.shape[0])))
    for i in range(starting_index,ending_index):
        rounded_empirical_risk = np.ceil(r_hats[i] * 10**sig_figs)/(10**sig_figs)#To make more efficient caching
        if nu_plus(loss_table.shape[0], m, rounded_empirical_risk, alpha, delta, 20, lambdas.shape[0]) < alpha:
            break
    for j in reversed(range(i,ending_index)):
        rounded_empirical_risk = np.ceil(r_hats[j] * 10**sig_figs)/(10**sig_figs)#To make more efficient caching
        if nu_plus(loss_table.shape[0], m, np.round(r_hats[j],sig_figs), alpha, delta, 20, lambdas.shape[0]) < alpha:
            break
    if i == j:
        R = np.array([])
    else:
        R = np.array([i,j]) 
    return R

"""
    SIMULATION OF LOSSES
"""
@cacheable
def get_empirical_process_means(mu_low, mu_high, num_mu):
    possible_mus = np.linspace(mu_low, mu_high, num_mu)
    return stats.norm.cdf(np.random.normal(size=(10000,num_mu)) + possible_mus[None,:]).mean(axis=0)

def get_process_mean_function():
    mu_low, mu_high, num_mu = (-20, 20, 10000)
    possible_mus = np.linspace(mu_low, mu_high, num_mu)
    empirical_means = get_empirical_process_means(mu_low,mu_high,num_mu)
    def _mean_function(mu):
        return empirical_means[np.abs(possible_mus-mu).argmin()]
    return _mean_function

def AR_Noise_Process(signal,alpha,n,N,corr,mean_function):
    sigma2 = 1/(1-corr**2)
    # Now find the sequence of mus that leads to the right expected values of an AR process
    mus = np.zeros_like(signal)
    for j in range(N):
        def _condition(mu_j):
            return mean_function(mu_j) - signal[j]
        mus[j] = brentq(_condition,-100,100)
    # Simulate the AR process now
    loss_table = np.zeros((n,N))
    u = np.random.normal(loc=0,scale=np.sqrt(sigma2),size=n)
    for j in range(N):
        loss_table[:,j] = stats.norm.cdf(u + mus[j])
        u = corr*u + np.sqrt(1-corr*corr)*np.random.normal(loc=0,scale=1,size=n)
    return loss_table

"""
    PLOT SIMULATION AND REJECTION REGIONS
"""

def get_simulation_and_rejection_regions(n,N,m,delta,alpha,corr,peak,downsample_factor):
    # Create a signal that dips below alpha at some points 
    signal = np.concatenate((np.linspace(peak,alpha/4,int(np.floor(N/2))),np.linspace(alpha/4,peak,int(np.ceil(N/2)))),axis=0)
    mean_function = get_process_mean_function()
    loss_table = AR_Noise_Process(signal,alpha,n,N,corr,mean_function)
    lambdas = np.linspace(0,1,N)
    # Get rejection regions for different methods
    R_naive = naive_rejection_region(loss_table,lambdas,alpha,delta)
    R_bonferroni_search_HB = bonferroni_search_HB(loss_table,lambdas,alpha,delta,downsample_factor)
    R_bonferroni_HB = bonferroni_HB(loss_table,lambdas,alpha,delta)
    R_uniform = uniform_region(loss_table,lambdas,alpha,delta,m)

    Rs = [R_naive, 
            R_bonferroni_search_HB,
            R_bonferroni_HB,
            R_uniform,
         ] 

    labels = (  "Empirical \nrisk" + r' < $\alpha$',
                'Fixed\nSequence',
                r'Bonferroni',
                r'Uniform'
             )

    colors = ('#C18268',
              '#B4926D',
              '#DAFFC1',
              '#4A7087',
              '#7E8F91'
             )
    
    for i in range(len(Rs)):
        if len(Rs[i])==0:
            Rs[i] = np.array([np.abs(lambdas-0.5).argmin(),])

    endpoints = lambdas[[R.max() for R in Rs]]
    
    return list(labels), endpoints.tolist(), list((alpha,)*len(labels))

if __name__ == "__main__":
    ns = [100, 500, 1000, 4000]
    N = 1000
    m = 1000
    delta = 0.1
    alphas = (0.1, 0.15, 0.2)
    # Define the correlation of the AR noise process
    corr = 0.9 
    peaks = (0.4,)
    downsample_factor = 10

    for n in ns:
        for peak in peaks:
            labels = []
            endpoints = []
            alpha_list = []
            for i in reversed(range(len(alphas))):
                out = get_simulation_and_rejection_regions(n,N,m,delta,alphas[i],corr,peak,downsample_factor)
                labels = labels + out[0]
                endpoints = endpoints + out[1]
                alpha_list = alpha_list + out[2]

            labels = np.array(labels)
            endpoints = np.array(endpoints)
            alpha_list = np.array(alpha_list)

            table_string = ""
            for label in np.unique(labels):
                table_string += f"{label} "
                for a in np.unique(alpha_list):
                    endpoint = endpoints[(labels==label) & (alpha_list==a)]
                    table_string += f"& {endpoint[0]:.2f}"
                table_string += "\\\\\n"

            table_file = open('../outputs/concentration_results/' + f'{str(peak)}_table'.replace(".","_") + f"_n={str(n)}" + '.txt', "w")
            n = table_file.write(table_string)
            table_file.close()

            # Now plot true signal
            plt.figure()
            i = -1
            signal = np.concatenate((np.linspace(peak,alphas[i]/4,int(np.floor(N/2))),np.linspace(alphas[i]/4,peak,int(np.ceil(N/2)))),axis=0)
            lambdas = np.linspace(0,1,N)
            plt.plot(lambdas,signal,alpha=1,color='k',linewidth=3, label="True Risk")
            #plt.plot(lambdas,loss_table.mean(axis=0),alpha=1,color='k',linewidth=3, label="Empirical Risk")
            plt.gca().axhline(alphas[i],xmin=min(lambdas),xmax=max(lambdas),linewidth=3,alpha=1,color='#888888',linestyle='dashed',label=r'$\alpha$')
            plt.gca().set_xlabel(r'$\lambda$',fontsize=20)
            plt.gca().set_ylabel('True Risk',fontsize=20)
            plt.gca().set_yticks([0,0.25,0.5,0.75,1])
            plt.gca().set_xticks([0,0.25,0.5,0.75,1])
            xticklabels = plt.gca().get_xticklabels()
            plt.setp(xticklabels, rotation=45, fontsize=15)
            yticklabels = plt.gca().get_yticklabels()
            plt.setp(yticklabels, fontsize=15)
            #plt.xlim(left=0.2,right=0.8)
            sns.despine(top=True,right=True)
            plt.tight_layout()
            plt.savefig(f"../outputs/concentration_results/{str(peak).replace('.','_')}_true_mean_n={str(n)}.pdf")
