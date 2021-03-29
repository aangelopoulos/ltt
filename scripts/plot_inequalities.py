import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import brentq
from concentration import *
import pdb

def plot_upper_tail(ns,s,ms,alpha,etas,maxiter):
    plt.figure()
    # Plot upper tail
    for m in ms:
        for eta in etas:
            shats = [shat_upper_tail(s, n, m, alpha, eta, maxiter) for n in ns]
            plt.plot(ns,shats,label=f'm={m},'+r'$\eta$' + f'={eta}')
    plt.xscale('log')
    plt.axhline(y=s, xmin=0, xmax=1, linestyle='dashed')
    plt.ylim([s-0.02,1])
    plt.legend()
    plt.savefig('../outputs/shat_upper_tail.pdf')

def plot_lower_tail(ns,s,ms,alpha,etas,maxiter):
    plt.figure()
    # Plot lower tail
    for m in ms:
        for eta in etas:
            shats = [shat_lower_tail(s, n, m, alpha, eta, maxiter) for n in ns]
            plt.plot(ns,shats,label=f'm={m},'+r'$\eta$' + f'={eta}')
    plt.xscale('log')
    plt.axhline(y=s, xmin=0, xmax=1, linestyle='dashed')
    plt.ylim([0,s+0.02])
    plt.legend()
    plt.savefig('../outputs/shat_lower_tail.pdf')

if __name__ == "__main__":
    ns = np.logspace(2,5,50)
    s = 0.2
    ms = [100, 1000, 10000]
    alpha = 0.2
    eta = [0.001, 0.01, 0.1]
    maxiter = 100000
    
    plot_lower_tail(ns,s,ms,alpha,eta,maxiter)
    plot_upper_tail(ns,s,ms,alpha,eta,maxiter)
