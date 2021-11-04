import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import brentq
from concentration import *
from uniform_concentration import *
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

def plot_required_fdp(ns, m, alphas, deltas, maxiter):
    sns.set(font_scale=1.5)
    columns = ['alpha_plus','n','m','alpha','delta']
    concat_list = []
    # Plot upper tail
    for i in range(len(alphas)):
        alpha = alphas[i]
        for j in range(len(deltas)):
            delta = deltas[j]
            local_list = [pd.DataFrame.from_dict({'alpha_plus': [required_empirical_risk(alpha, n, m, alpha, delta, maxiter, 100),],
                                                  'n': [n,],
                                                  'm': [m,],
                                                  'alpha': [alpha,],
                                                  'delta': [delta,]}) 
                         for n in ns]
            concat_list = concat_list + local_list
    df = pd.concat(concat_list, ignore_index=True)

    with sns.axes_style('white'):
        g = sns.FacetGrid(df, row='alpha', col='delta', margin_titles=True, height=2.5, sharex=False, sharey=False)
        g.map(sns.lineplot, 'n', 'alpha_plus', color='#334488', linewidth=2)
        g.map(sns.lineplot, 'n', 'alpha', color='#888888', alpha=0.8, linewidth=2)
        g.set_axis_labels('n', 'Required FDP')
        g.set(xscale='log')
        g.fig.subplots_adjust(wspace=.2, hspace=.2)

    for i in range(len(alphas)):
        for j in range(len(deltas)):
            #g.axes[i,j].xaxis.label.set_size(15)
            #g.axes[i,j].yaxis.label.set_size(15)
            #g.axes[i,j].tick_params(axis='both', which='minor', labelsize=15)
            g.axes[i,j].xaxis.set_ticks([1e3,1e5])

            if i < len(alphas)-1 and j == 0:
                g.axes[i,j].xaxis.set_ticklabels([])
                continue
            if i == len(alphas)-1:
                if j > 0:
                    g.axes[i,j].yaxis.set_ticklabels([])
                continue
            g.axes[i,j].xaxis.set_ticklabels([])
            g.axes[i,j].yaxis.set_ticklabels([])

    plt.savefig('../outputs/concentration_results/alpha_plus.pdf')


if __name__ == "__main__":
    ns = np.logspace(2,5,50)
    m = 100
    deltas = [0.001, 0.01, 0.1, 0.2]
    deltas.reverse()
    alphas = [0.1, 0.2, 0.5, 0.7]
    alphas.reverse()
    maxiter = 1000
    
    plot_required_fdp(ns,m,alphas,deltas,maxiter)
