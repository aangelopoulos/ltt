import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import brentq
import pdb
from pathlib import Path
import pickle as pkl
from utils import *
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

def romano_wolf_multiplier_bootstrap(loss_table,lambdas,alpha,delta,B=1000):
    n = loss_table.shape[0]
    r_hats = loss_table.mean(axis=0) # empirical risk at each lambda
    z_table = np.zeros_like(loss_table)


"""
    NAIVE ALGORITHM
"""
# Just select the set of lambdas where the 1-delta quantile of the loss table is below alpha.
def naive_rejection_region(loss_table,lambdas,alpha,delta):
    quantiles = np.quantile(loss_table,1-delta,axis=0,interpolation='higher') 
    R = np.nonzero(quantiles < alpha)[0] 
    return R

"""
    SIMULATION OF LOSSES
"""
def AR_Noise_Process(signal,alpha,n,N):
    # Define the correlation of the AR noise process
    corr = 0.9
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

if __name__ == "__main__":
    N = 1000
    n = 4000
    m = 1000
    delta = 0.2
    alpha = 0.1
    # Create a signal that dips below alpha at some points 
    signal = np.concatenate((np.linspace(alpha*1.5,alpha/2,int(np.floor(N/2))),np.linspace(alpha/2,alpha*1.5,int(np.ceil(N/2)))),axis=0)
    loss_table = AR_Noise_Process(signal,alpha,n,N)
    lambdas = np.linspace(0,1,N)
    # Get rejection regions for different methods
    R_widest = (np.nonzero(signal < alpha)[0][0],np.nonzero(signal<alpha)[0][-1])
    R_naive = naive_rejection_region(loss_table,lambdas,alpha,delta)

    R = romano_wolf_HB(loss_table,lambdas,alpha,delta)
    
    plt.figure()
    plt.plot(lambdas,signal,alpha=1,color='k',linewidth=3, label="True Risk")
    plt.hlines(alpha,xmin=min(lambdas),xmax=max(lambdas),color='k',linewidth=3,alpha=0.5,linestyle='dashed',label=r'$\alpha$')
    plt.plot(lambdas,loss_table[0:10,:].T,alpha=0.05,color='#73D673') # Sample losses
    # Sets
    plt.hlines(0,xmin=lambdas[R_widest[0]],xmax=lambdas[R_widest[-1]],color='y',linewidth=3,label=r'Empirical risk < $\alpha$')
    plt.vlines((lambdas[R_widest[0]],lambdas[R_widest[-1]]),ymin=-0.02,ymax=0.02,color='y',linewidth=3)
    plt.hlines(0,xmin=lambdas[R_naive[0]],xmax=lambdas[R_naive[-1]],color='b',linewidth=3,label=r'Empirical risk < $\alpha$, with proportion 1-$\delta$')
    plt.vlines((lambdas[R_naive[0]],lambdas[R_naive[-1]]),ymin=-0.02,ymax=0.02,color='b',linewidth=3)
    plt.hlines(0,xmin=lambdas[min(R)],xmax=lambdas[max(R)],color='r',linewidth=3,label=r'RWHB Rejection Region')
    plt.vlines((lambdas[min(R)],lambdas[max(R)]),ymin=-0.02,ymax=0.02,color='r',linewidth=3)
    # Finish
    plt.legend()
    plt.show()


    pdb.set_trace()
    print("HI")

