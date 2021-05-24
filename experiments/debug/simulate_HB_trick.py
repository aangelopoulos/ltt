import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.bounds import HB_mu_plus
import pdb

if __name__ == "__main__":
    n = 30000
    fdrs = np.logspace(-2,0,100)
    alpha = 0.2
    delta = 0.1
    abstention_frequency = 0.6
    maxiters = 1000
    true_means = np.zeros_like(fdrs)
    fdrpluses = np.zeros_like(fdrs)
    for i in range(fdrs.shape[0]):
        fdr = fdrs[i]
        pvals = np.array([(1-fdr)*(1-abstention_frequency),abstention_frequency,fdr*(1-abstention_frequency)])
        true_means[i] = (pvals * np.array([0,alpha,1])).sum()
        mu = np.random.multinomial(n,pvals).dot(np.array([0,alpha,1]))/n
        fdrpluses[i] = HB_mu_plus(mu, n, delta, maxiters) 
    plt.plot(fdrs, fdrpluses-true_means, color='y', linewidth=3)
    #plt.plot(fdrs, true_means, color='k',linewidth=3)
    plt.ylim([0,None])
    sns.despine(top=True,right=True)
    plt.savefig("./outputs/HB_gap.pdf")
