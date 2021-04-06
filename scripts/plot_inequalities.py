import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import brentq
from concentration import *
import pdb

def plot_upper_tail(ns,s,ms,delta,maxiter):
    plt.figure()
    # Plot upper tail
    for m in ms:
        eta_star_upper = get_eta_star_upper(ns[0], m, alpha, delta, maxiter)
        shats = [shat_upper_tail(s, n, m, delta, eta_star_upper, maxiter) for n in ns]
        plt.plot(ns,shats,label=f'm={m}')
    plt.xscale('log')
    plt.axhline(y=s, xmin=0, xmax=1, linestyle='dashed')
    plt.ylim([s-0.02,1])
    plt.legend()
    plt.savefig('../outputs/shat_upper_tail.pdf')

def plot_required_fdp(ns,m,alphas,deltas,maxiter):
    fig, axs = plt.subplots(nrows=len(alphas), ncols=len(deltas))

    # Plot upper tail
    for i in range(len(alphas)):
        alpha = alphas[i]
        for j in range(len(deltas)):
            delta = deltas[j]
            eta_star_upper = get_eta_star_upper(ns[0], m, alpha, delta, maxiter)
            alpha_pluses = [required_fdp(n, m, alpha, delta, maxiter) for n in ns]
            axs[i,j].plot(ns,alpha_pluses)
    plt.xscale('log')
    plt.axhline(y=alpha, xmin=0, xmax=1, linestyle='dashed')
    plt.ylim([0,1.02 * alpha])
    plt.tight_layout()
    plt.savefig('../outputs/alpha_plus.pdf')

def plot_required_fdp_dataframe(ns, m, alphas, deltas, maxiter):
    columns = ['alpha_plus','n','m','alpha','delta']
    concat_list = []
    # Plot upper tail
    for i in range(len(alphas)):
        alpha = alphas[i]
        for j in range(len(deltas)):
            delta = deltas[j]
            eta_star_upper = get_eta_star_upper(ns[0], m, alpha, delta, maxiter)
            local_list = [pd.DataFrame.from_dict({'alpha_plus': [required_fdp(n, m, alpha, delta, maxiter),],
                                                  'n': [n,],
                                                  'm': [m,],
                                                  'alpha': [alpha,],
                                                  'delta': [delta,]}) 
                         for n in ns]
            concat_list = concat_list + local_list
    df = pd.concat(concat_list, ignore_index=True)

    with sns.axes_style('white'):
        g = sns.FacetGrid(df, row='alpha', col='delta', margin_titles=True, height=2.5, sharex=False, sharey=False)
        g.map(sns.lineplot, 'n', 'alpha_plus', color='#334488')
        g.map(sns.lineplot, 'n', 'alpha', color='#888888', alpha=0.8)
        g.set_axis_labels('n', 'Required FDP')
        g.set(xscale='log')
        #g.set(xscale='log')
        g.fig.subplots_adjust(wspace=.2, hspace=.2)

    plt.savefig('../outputs/alpha_plus.pdf')


if __name__ == "__main__":
    ns = np.logspace(2,5,50)
    m = 100
    deltas = [0.001, 0.01, 0.1, 0.2]
    deltas.reverse()
    alphas = [0.1, 0.2, 0.5, 0.7]
    alphas.reverse()
    maxiter = 1000
    
    plot_required_fdp_dataframe(ns,m,alphas,deltas,maxiter)
