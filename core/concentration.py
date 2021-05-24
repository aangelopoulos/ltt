import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
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
from utils import *
from core.uniform_concentration import nu_plus
from core.pfdr import *
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
def hb_p_value(r_hat,n,alpha):
    bentkus_p_value = np.e * stats.binom.cdf(np.ceil(n*r_hat),n,alpha)
    def h1(y,mu):
        return y * np.log(y/mu) + (1-y) * np.log((1-y)/(1-mu))
    hoeffding_p_value = np.exp(-n*h1(min(r_hat,alpha),alpha))
    return min(bentkus_p_value,hoeffding_p_value)

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
    BONFERRONI SPECIALIZATIONS 
"""
def bonferroni_HB(loss_table,lambdas,alpha,delta):
    n = loss_table.shape[0]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    return bonferroni(p_values,delta)

def bonferroni_CLT(loss_table,lambdas,alpha,delta):
    t_values = np.sqrt(loss_table.shape[0])*(alpha - loss_table.mean(axis=0))/loss_table.std(axis=0) 
    p_values = 1-stats.norm.cdf(t_values)
    return bonferroni(p_values,delta)

"""
    BONFERRONI SEARCH MASTER ALGORITHM
"""
def bonferroni_search(p_values,delta,downsample_factor):
    N = p_values.shape[0]
    N_coarse = int(p_values.shape[0]/downsample_factor)
    # Downsample, making sure to include the endpoints.
    coarse_indexes = set(range(0,N,downsample_factor))
    coarse_indexes.update({0,N-1})
    R = set() 
    for idx in coarse_indexes:
        _idx = idx
        while _idx < N and p_values[_idx] < delta/N_coarse:
            R.update({_idx})
            _idx = _idx - 1
    return np.array(list(R))

"""
    BONFERRONI SEARCH SPECIALIZATIONS 
"""
def bonferroni_search_HB(loss_table,lambdas,alpha,delta,downsample_factor):
    n = loss_table.shape[0]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    return bonferroni_search(p_values,delta,downsample_factor)

def bonferroni_search_CLT(loss_table,lambdas,alpha,delta,downsample_factor):
    t_values = np.sqrt(loss_table.shape[0])*(alpha - loss_table.mean(axis=0))/loss_table.std(axis=0) 
    p_values = 1-stats.norm.cdf(t_values)
    return bonferroni_search(p_values,delta,downsample_factor)

"""
    NAIVE ALGORITHM
"""
# Just select the set of lambdas where the 1-delta quantile of the loss table is below alpha.
def naive_rejection_region(loss_table,lambdas,alpha,delta):
    return np.array([np.nonzero(loss_table.mean(axis=0) < alpha)[0][0],np.nonzero(loss_table.mean(axis=0)<alpha)[0][-1]])

"""
    UNIFORM REGION (only returns endpoints) 
"""
def uniform_region(loss_table,lambdas,alpha,delta,m):
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda (FDP)
    starting_index = (r_hats < alpha).nonzero()[0][0]
    ending_index = (r_hats < alpha).nonzero()[0][-1]
    R = np.array([])
    sig_figs = int(np.ceil(np.log10(lambdas.shape[0])))
    for i in range(starting_index,ending_index):
        rounded_empirical_risk = np.ceil(r_hats[i] * 10**sig_figs)/(10**sig_figs)#To make more efficient caching
        if nu_plus(loss_table.shape[0], m, rounded_empirical_risk, delta, 20, lambdas.shape[0]) < alpha:
            break
    for j in reversed(range(i,ending_index)):
        rounded_empirical_risk = np.ceil(r_hats[j] * 10**sig_figs)/(10**sig_figs)#To make more efficient caching
        if nu_plus(loss_table.shape[0], m, np.round(r_hats[j],sig_figs), delta, 20, lambdas.shape[0]) < alpha:
            break
    R = np.array([i,j]) 
    return R

"""
    SIMULATION OF LOSSES
"""
def AR_Noise_Process(signal,alpha,n,N,corr):
    sigma2 = 1/(1-corr**2)
    # Now find the sequence of mus that leads to the right expected values of an AR process
    mus = np.zeros_like(signal)
    for j in range(N):
        def _condition(mu_j):
            return stats.norm.cdf(0,-mu_j,np.sqrt(1+sigma2)) - signal[j]
        mus[j] = brentq(_condition,-100,100)
    # Simulate the AR process now
    loss_table = np.zeros((n,N))
    u = np.random.normal(loc=0,scale=np.sqrt(sigma2),size=n)
    for j in range(N):
        loss_table[:,j] = stats.norm.cdf(u + mus[j])
        u = corr*u + np.random.normal(loc=0,scale=1,size=n)
    return loss_table

"""
    PLOT SIMULATION AND REJECTION REGIONS
"""

def plot_simulation_and_rejection_regions(ax,n,N,m,delta,alpha,corr,peak,downsample_factor):
    # Create a signal that dips below alpha at some points 
    signal = np.concatenate((np.linspace(peak,alpha/4,int(np.floor(N/2))),np.linspace(alpha/4,peak,int(np.ceil(N/2)))),axis=0)
    loss_table = AR_Noise_Process(signal,alpha,n,N,corr)
    lambdas = np.linspace(0,1,N)
    # Get rejection regions for different methods
    R_naive = naive_rejection_region(loss_table,lambdas,alpha,delta)
    R_RW_bootstrap = romano_wolf_multiplier_bootstrap(loss_table,lambdas,alpha,delta)
    R_RW_HB = romano_wolf_HB(loss_table,lambdas,alpha,delta)
    R_RW_CLT = romano_wolf_CLT(loss_table,lambdas,alpha,delta)
    R_bonferroni_search_HB = bonferroni_search_HB(loss_table,lambdas,alpha,delta,downsample_factor)
    R_bonferroni_HB = bonferroni_HB(loss_table,lambdas,alpha,delta)
    R_bonferroni_CLT = bonferroni_CLT(loss_table,lambdas,alpha,delta)
    R_uniform = uniform_region(loss_table,lambdas,alpha,delta,m)

    Rs = (R_naive, 
            R_RW_bootstrap, 
            R_RW_HB, 
            R_RW_CLT, 
            R_bonferroni_search_HB,
            R_bonferroni_HB,
            R_bonferroni_CLT,
            R_uniform)

    labels = (r'Empirical risk < $\alpha$',
                r'RWMB Rejections',
                r'RWHB Rejections',
                r'RWCLT Rejections',
                r'BonferroniSearchHB Rejections',
                r'BonferroniHB Rejections',
                r'BonferroniCLT Rejections',
                r'Bardenet Rejections (uniform)')

    colors = ('#C18268',
              '#B4926D',
              '#C1DAFF',
              '#DAFFC1',
              '#FFDAC1',
              '#4A7087',
              '#6D92B4',
              '#887D82')
    
    #ax.plot(lambdas,loss_table[0:8,:].T,alpha=0.3,color='#73D673') # Sample losses
    ax.plot(lambdas,signal,alpha=1,color='k',linewidth=3, label="True Risk")
    ax.axhline(alpha,xmin=min(lambdas),xmax=max(lambdas),linewidth=3,alpha=1,color='#888888',linestyle='dashed',label=r'$\alpha$')

    # Sets
    for i in range(len(Rs)):
        if len(Rs[i]) == 0:
            print("Empty region:" + labels[i])
        else:
            ax.hlines(-0.04*(i+1),xmin=lambdas[min(Rs[i])],xmax=lambdas[max(Rs[i])],linewidth=3,color=colors[i],label=labels[i])
            ax.vlines((lambdas[min(Rs[i])],lambdas[max(Rs[i])]),ymin=-0.04*(i+1)-0.02,ymax=-0.04*(i+1)+0.02,linewidth=3,color=colors[i])

    # Finish
    sns.despine(top=True,right=True)

if __name__ == "__main__":
    n = 4000
    N = 1000
    m = 1000
    delta = 0.1
    alphas = (0.1, 0.15, 0.2)
    # Define the correlation of the AR noise process
    corrs = (0.99, 0.95, 0.90)
    peaks = (0.8,0.4)
    downsample_factor = 10

    for peak in peaks:
        fig, axs = plt.subplots(nrows=len(alphas), ncols=len(corrs), sharex=True, sharey=True, figsize=(len(alphas)*4,len(corrs)*4))
        for i in reversed(range(len(alphas))):
            for j in reversed(range(len(corrs))):
                plot_simulation_and_rejection_regions(axs[i,j],n,N,m,delta,alphas[i],corrs[j],peak,downsample_factor)
                if i == 0:
                    axs[i,j].set_title("corr=" + str(corrs[j]), fontsize=20)
                if j == 0:
                    axs[i,j].set_ylabel(r"$\alpha=$" + str(alphas[i]), fontsize=20)
                axs[i,j].set_xticks([.2,.5,.8])
                axs[i,j].set_xticklabels([.2,.5,.8], fontsize=15)
                axs[i,j].set_yticks([0,.25,.5])
                axs[i,j].set_yticklabels([0,.25,.5], fontsize=15)

        if peak == peaks[-1]:
            axs[len(alphas)-1,len(corrs)-1].legend(loc='upper right', fontsize=15)
        plt.xlim(left=0.2,right=0.8)
        plt.savefig(f"../outputs/concentration_results/{str(peak).replace('.','_')}_concentration_comparison.pdf")
