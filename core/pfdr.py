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
from core.uniform_concentration import nu_plus, r_minus 
from core.concentration import * 
from core.bounds import binom_p_value, HB_mu_plus, HB_mu_minus
from tqdm import tqdm
CACHE = str(Path(__file__).parent.absolute()) + '/.cache/'

def get_selective_risk_p_values(score_vector, correct_vector, lambdas, alpha):
    # Define selective risk
    def selective_risk(lam): return (1-correct_vector[score_vector > lam]).sum()/(score_vector  > lam).sum()
    def nlambda(lam): return (score_vector  > lam).sum()
    cut_lambdas = np.array([lam for lam in lambdas if nlambda(lam) >= 25]) # Make sure there's some data in the top bin.
    p_values = np.array([binom_p_value(selective_risk(lam),nlambda(lam),alpha) for lam in cut_lambdas])
    return p_values, cut_lambdas

def get_nus_rs_n(score_vector, correct_vector, lambdas):
    try:
        score_vector = score_vector.numpy()
        correct_vector = correct_vector.numpy()
    except:
        # already numpy
        pass
    n = score_vector.shape[0]
    nus = np.array([(1-correct_vector)[score_vector > lam].astype(float).sum()/n for lam in lambdas])
    nus = np.nan_to_num(nus)
    rs = np.array([(score_vector > lam).astype(float).mean() for lam in lambdas])
    rs = np.nan_to_num(rs)
    return nus, rs, n

def pfdr_loss_table(score_vector, correct_vector, lambdas, alpha):
    n = score_vector.shape[0]
    N = lambdas.shape[0]
    loss_table = np.zeros((n,N))
    for j in range(N):
        predict = score_vector > lambdas[j]
        c = correct_vector
        loss_table[:,j] = (1-c)*predict - alpha * predict + alpha 
    return loss_table

"""
    BONFERRONI SPECIALIZATIONS 
"""

def pfdr_bonferroni_binom(score_vector,correct_vector,lambdas,alpha,delta):
    p_values, _ = get_selective_risk_p_values(score_vector,correct_vector,lambdas,alpha)
    p_values = np.nan_to_num(p_values, nan=1.0)
    return bonferroni(p_values,delta)

def pfdr_uniform(score_vector,correct_vector,lambdas,alpha,delta,m=1000,maxiter=1000,num_grid_points=None):
    nus, rs, n = get_nus_rs_n(score_vector, correct_vector, lambdas)
    N = lambdas.shape[0]
    s_arr = nus - alpha * rs + alpha
    # subset only to search the ones with low enough s
    starting_index = (s_arr < alpha).nonzero()[0][0]
    ending_index = (s_arr < alpha).nonzero()[0][-1]

    upper_bounds_arr = np.array([ nu_plus(n, m, s, alpha, delta, maxiter, num_grid_points) for s in s_arr[starting_index:min((ending_index+1),N)] ])
    R = np.nonzero(upper_bounds_arr < alpha)[0] + starting_index
    return R

"""
    BONFERRONI SEARCH SPECIALIZATIONS 
"""
def pfdr_bonferroni_search_binom(score_vector, correct_vector, lambdas, alpha, delta, downsample_factor=10):
    p_values, _ = get_selective_risk_p_values(score_vector,correct_vector,lambdas,alpha)
    R = lambdas.shape[0] -  bonferroni_search(p_values[::-1],delta,lambdas.shape[0])[::-1] - 1
    return R 

if __name__ == "__main__":
    n = 1000
    cutoff = 0.7
    score_vector = np.random.uniform(size=(n,)) 
    correct_vector = score_vector > cutoff
    lambdas = np.linspace(0,1,20)
    alpha = 0.1
    delta = 0.1
    R = pfdr_bonferroni_HB(score_vector, correct_vector, lambdas, alpha, delta)
    print("Hello, World!")

