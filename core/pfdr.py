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
from core.bounds import hb_p_value, HB_mu_plus, HB_mu_minus
from tqdm import tqdm
CACHE = str(Path(__file__).parent.absolute()) + '/.cache/'

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
    RW SPECIALIZATIONS
"""

def pfdr_romano_wolf_multiplier_bootstrap(score_vector, correct_vector, lambdas, alpha, delta, B=100): 
    n = score_vector.shape[0]
    N = lambdas.shape[0]
    loss_table = pfdr_loss_table(score_vector, correct_vector, lambdas, alpha)
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

# correct_vector_i = 1(top class is correct on example i)
# score_vector_i = score of top class for example i
def pfdr_romano_wolf_HB(score_vector,correct_vector,lambdas,alpha,delta):
    nus, rs, n = get_nus_rs_n(score_vector, correct_vector, lambdas)
    r_hats = nus - alpha*rs + alpha
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    p_values = np.nan_to_num(p_values, nan=1.0)
    def subset_scoring_function(S):
        return delta/len(S)
    #return np.nonzero(nus - alpha * rs + alpha < alpha)[0]
    return romano_wolf(p_values,subset_scoring_function)

"""
    BONFERRONI SPECIALIZATIONS 
"""

# correct_vector_i = 1(top class is correct on example i)
# score_vector_i = score of top class for example i
def pfdr_bonferroni_HB(score_vector,correct_vector,lambdas,alpha,delta):
    nus, rs, n = get_nus_rs_n(score_vector, correct_vector, lambdas)
    r_hats = nus-alpha*rs+alpha
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    p_values = np.nan_to_num(p_values, nan=1.0)
    return bonferroni(p_values,delta)

def pfdr_ucb_HB(n, nu, r, delta, maxiter):
    nu_p = HB_mu_plus(nu, n, delta, maxiter)
    r_m = HB_mu_minus(r, n, delta, maxiter)
    if r_m <= 0 and nu_p > 0:
        return np.Inf 
    if nu_p <= 0:
        return 0 
    return nu_p/r_m

def pfdr_HB(score_vector, correct_vector, lambdas, alpha, delta, m=1000, maxiter=1000):
    nus, rs, n = get_nus_rs_n(score_vector, correct_vector, lambdas)
    starting_index = (nus/rs < alpha).nonzero()[0][0]

    pfdr_pluses = torch.tensor( [ pfdr_ucb_HB(n, nus[i], rs[i], delta, maxiter) for i in range(starting_index, calib_nu.shape[0]) ] )

    if ((pfdr_pluses > alpha).float().sum() == 0):
        valid_set_index = 0
    else:
        valid_set_index = max((pfdr_pluses > alpha).nonzero()[0][0]+starting_index-1, 0)  # -1 because it needs to be <= alpha
    
    R = np.array([valid_set_index,])
    return R

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
def pfdr_bonferroni_search_HB(score_vector, correct_vector, lambdas, alpha, delta, downsample_factor=10):
    nus, rs, n = get_nus_rs_n(score_vector, correct_vector, lambdas)
    N = lambdas.shape[0]
    r_hats = [ nus[i]-alpha*rs[i]+alpha for i in range(len(nus)) ] # using lihua's note, empirical risk at each lambda
    p_values = np.array([hb_p_value(r_hat,n,alpha) for r_hat in r_hats])
    p_values = np.nan_to_num(p_values, nan=1.0)
    p_values[-1] = 0.0
    R = N-bonferroni_search(p_values[::-1],delta,downsample_factor)-1
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

