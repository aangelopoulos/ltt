import os
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
from uniform_concentration import required_fdp
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

def romano_wolf_multiplier_bootstrap(loss_table,lambdas,alpha,delta,B=500):
    n = loss_table.shape[0]
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
    BONFERRONI SPECIALIZATIONS 
"""
def bonferroni_HB(loss_table,lambdas,alpha,delta):
    n = loss_table.shape[0]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    rejections, _, _, _ = multipletests(p_values,delta,method='holm',is_sorted=False,returnsorted=False)
    R = np.nonzero(rejections)[0]
    return R 

"""
    NAIVE ALGORITHM
"""
# Just select the set of lambdas where the 1-delta quantile of the loss table is below alpha.
def naive_rejection_region(loss_table,lambdas,alpha,delta):
    quantiles = np.quantile(loss_table,1-delta,axis=0,interpolation='higher') 
    R = np.nonzero(quantiles < alpha)[0] 
    return R

"""
    UNIFORM REGION 
"""
def uniform_region(loss_table,lambdas,alpha,delta,m):
    thresh = required_fdp(loss_table.shape[0], m, alpha, delta, maxiter=100,num_grid_points=lambdas.shape[0])
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda (FDP)
    R = np.nonzero(r_hats < thresh)[0]
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

def plot_simulation_and_rejection_regions(ax,n,N,m,delta,alpha,corr):
    # Create a signal that dips below alpha at some points 
    signal = np.concatenate((np.linspace(alpha*1.5,alpha/4,int(np.floor(N/2))),np.linspace(alpha/4,alpha*1.5,int(np.ceil(N/2)))),axis=0)
    loss_table = AR_Noise_Process(signal,alpha,n,N,corr)
    lambdas = np.linspace(0,1,N)
    # Get rejection regions for different methods
    R_widest = (np.nonzero(signal < alpha)[0][0],np.nonzero(signal<alpha)[0][-1])
    R_RW_bootstrap = romano_wolf_multiplier_bootstrap(loss_table,lambdas,alpha,delta)
    R_RW_HB = romano_wolf_HB(loss_table,lambdas,alpha,delta)
    R_bonferroni_HB = bonferroni_HB(loss_table,lambdas,alpha,delta)
    R_naive = naive_rejection_region(loss_table,lambdas,alpha,delta)
    R_uniform = uniform_region(loss_table,lambdas,alpha,delta,m)

    Rs = (R_widest, 
            R_naive,
            R_RW_bootstrap, 
            R_RW_HB, 
            R_bonferroni_HB,
            R_uniform)

    labels = (r'Empirical risk < $\alpha$',
                r'1-$\delta$ proportion of losses < $\alpha$',
                r'RWMB Rejections',
                r'RWHB Rejections',
                r'HBBonferroni Rejections',
                r'Bardenet Rejections (uniform)')

    colors = ('#C18268',
              '#5F9A84',
              '#B4926D',
              '#FFDAC1',
              '#4A7087',
              '#887D82')
    
    ax.plot(lambdas,loss_table[0:8,:].T,alpha=0.1,color='#73D673') # Sample losses
    ax.plot(lambdas,signal,alpha=1,color='k',linewidth=3, label="True Risk")
    ax.axhline(alpha,xmin=min(lambdas),xmax=max(lambdas),linewidth=3,alpha=1,color='#888888',linestyle='dashed',label=r'$\alpha$')

    # Sets
    for i in range(len(Rs)):
        if len(Rs[i]) == 0:
            print("Empty region:" + labels[i])
        else:
            ax.axhline(-0.04*i,xmin=lambdas[Rs[i][0]],xmax=lambdas[Rs[i][-1]],linewidth=3,color=colors[i],label=labels[i])
            #ax.axvline(lambdas[Rs[i][0]],ymin=-0.04*i-0.02,ymax=-0.04*i+0.02,color=colors[i],linewidth=3)
            #ax.axvline(lambdas[Rs[i][-1]],ymin=-0.04*i-0.02,ymax=-0.04*i+0.02,color=colors[i],linewidth=3)

    # Finish
    sns.despine(top=True,right=True)

if __name__ == "__main__":
    n = 5000
    N = 1000
    m = 1000
    delta = 0.1
    alphas = (0.05, 0.1, 0.15)
    # Define the correlation of the AR noise process
    corrs = (0.01, 0.1, 0.9)
    fig, axs = plt.subplots(nrows=len(alphas), ncols=len(corrs), sharex=True, sharey=True, figsize=(len(alphas)*4,len(corrs)*4))

    for i in range(len(alphas)):
        for j in range(len(corrs)):
            plot_simulation_and_rejection_regions(axs[i,j],n,N,m,delta,alphas[i],corrs[j])
            if i == 0:
                axs[i,j].set_title("corr=" + str(corrs[j]))
            if j == 0:
                axs[i,j].set_ylabel(r"$\alpha=$" + str(alphas[i]))

    axs[len(alphas)-1,len(corrs)-1].legend()
    plt.savefig("../outputs/concentration_results/concentration_comparison.pdf")
