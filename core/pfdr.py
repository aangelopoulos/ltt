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
from core.bounds import HB_mu_plus, HB_mu_minus
from tqdm import tqdm
CACHE = str(Path(__file__).parent.absolute()) + '/.cache/'

"""
    RW SPECIALIZATIONS
"""
# correct_vector_i = 1(top class is correct on example i)
# score_vector_i = score of top class for example i
def romano_wolf_HB(score_vector,correct_vector,lambdas,alpha,delta):
    n = score_vector.shape[0]
    nus = [1-correct_vector[score_vector > lam].astype(float).mean() for lam in lambdas]
    rs = [(score_vector > lam).astype(float).mean() for lam in lambdas]
    r_hats = [ (nus[i]-alpha*rs[i]+alpha)/(1+alpha) for i in range(len(nus)) ] # using lihua's note, empirical risk at each lambda (adjusted)
    alpha_adjusted = alpha/(1+alpha)
    p_values = np.array([hb_p_value(r_hat,n,alpha_adjusted) for r_hat in r_hats])
    p_values = np.nan_to_num(p_values, nan=1.0)
    def subset_scoring_function(S):
        return delta/len(S)
    return romano_wolf(p_values,subset_scoring_function)

"""
    BONFERRONI SPECIALIZATIONS 
"""

# correct_vector_i = 1(top class is correct on example i)
# score_vector_i = score of top class for example i
def pfdr_bonferroni_HB(score_vector,correct_vector,lambdas,alpha,delta):
    n = correct_vector.shape[0]
    nus = [1-correct_vector[score_vector > lam].astype(float).mean() for lam in lambdas]
    rs = [(score_vector > lam).astype(float).mean() for lam in lambdas]
    r_hats = [ (nus[i]-alpha*rs[i]+alpha)/(1+alpha) for i in range(len(nus)) ] # using lihua's note, empirical risk at each lambda (adjusted)
    alpha_adjusted = alpha/(1+alpha)
    p_values = np.array([hb_p_value(r_hat,n,alpha_adjusted) for r_hat in r_hats])
    p_values = np.nan_to_num(p_values, nan=1.0)
    return bonferroni(p_values,delta)

def pfdr_ucb_HB(num_calib, accuracy, frac_abstention, delta, maxiter):
    nu_p = HB_mu_plus(1-accuracy, num_calib, delta, maxiter)
    r_m = HB_mu_minus(1-frac_abstention, num_calib, delta, maxiter)
    if r_m <= 0 and nu_p > 0:
        return np.Inf 
    if nu_p <= 0:
        return 0 
    return nu_p/r_m

def pfdr_HB(score_vector, correct_vector, lambdas, alpha, delta, m=1000, maxiter=1000):
    num_calib = score_vector.shape[0]
    calib_accuracy = torch.tensor([ correct_vector[score_vector > lam].astype(float).mean() for lam in lambdas ])
    calib_abstention_freq = torch.tensor([ 1-(score_vector > lam).astype(float).mean() for lam in lambdas ])
    starting_index = ((1-calib_accuracy)/(1-calib_abstention_freq) < alpha).nonzero()[0][0]

    pfdr_pluses = torch.tensor( [ pfdr_ucb_HB(num_calib, calib_accuracy[i], calib_abstention_freq[i], delta, maxiter) for i in range(starting_index, calib_accuracy.shape[0]) ] )

    if ((pfdr_pluses > alpha).float().sum() == 0):
        valid_set_index = 0
    else:
        valid_set_index = max((pfdr_pluses > alpha).nonzero()[0][0]+starting_index-1, 0)  # -1 because it needs to be <= alpha
    
    R = np.array([valid_set_index,])
    return R

def pfdr_uniform(score_vector,correct_vector,lambdas,alpha,delta,m=1000,maxiter=1000,num_grid_points=None):
    num_calib = score_vector.shape[0]
    N = lambdas.shape[0]
    accuracy = np.array([ correct_vector[score_vector > lam].astype(float).mean() for lam in lambdas ])
    frac_abstention = np.array([ 1-(score_vector > lam).astype(float).mean() for lam in lambdas ])
    nu_arr = 1-accuracy
    r_arr = 1-frac_abstention
    s_arr = (nu_arr - alpha * r_arr + alpha)/(1+alpha)
    alpha_adjusted = alpha/(1+alpha)
    # subset only to search the ones with low enough s
    starting_index = (s_arr < alpha_adjusted).nonzero()[0][0]
    ending_index = (s_arr < alpha_adjusted).nonzero()[0][-1]

    upper_bounds_arr = np.array([ nu_plus(num_calib, m, s, delta, maxiter, num_grid_points) for s in s_arr[starting_index:min((ending_index+1),N)] ])
    R = np.nonzero(upper_bounds_arr < alpha_adjusted)[0] 
    return R

def pfdr_ucb_uniform_notrick(n, m, accuracy, frac_abstention, delta, maxiter, num_grid_points=None):
    nu_p = nu_plus(n, m, 1-accuracy, delta, maxiter, num_grid_points)
    r_m = r_minus(n, m, 1-frac_abstention, delta, maxiter,num_grid_points)
    if r_m <= 0 and nu_p > 0:
        return np.Inf 
    if nu_p <= 0:
        return 0 
    return nu_p/r_m

def pfdr_uniform_notrick(score_vector, correct_vector, lambdas, alpha, delta, m=1000, maxiter=1000):
    num_calib = score_vector.shape[0]
    calib_accuracy = torch.tensor([ correct_vector[score_vector > lam].astype(float).mean() for lam in lambdas ])
    calib_abstention_freq = torch.tensor([ 1-(score_vector > lam).astype(float).mean() for lam in lambdas ])
    starting_index = ((1-calib_accuracy)/(1-calib_abstention_freq) < alpha).nonzero()[0][0]

    pfdr_pluses = torch.tensor( [ pfdr_ucb_uniform_notrick(num_calib, m, calib_accuracy[i], calib_abstention_freq[i], delta, maxiter) for i in range(starting_index, calib_accuracy.shape[0]) ] )

    if ((pfdr_pluses > alpha).float().sum() == 0):
        valid_set_index = 0
    else:
        valid_set_index = max((pfdr_pluses > alpha).nonzero()[0][0]+starting_index-1, 0)  # -1 because it needs to be <= alpha
    
    R = np.array([valid_set_index,])
    return R

"""
    BONFERRONI SEARCH SPECIALIZATIONS 
"""
def pfdr_bonferroni_search_HB(score_vector, correct_vector, lambdas, alpha, delta, downsample_factor=10):
    n = correct_vector.shape[0]
    nus = [1-correct_vector[score_vector > lam].astype(float).mean() for lam in lambdas]
    rs = [(score_vector > lam).astype(float).mean() for lam in lambdas]
    r_hats = [ (nus[i]-alpha*rs[i]+alpha)/(1+alpha) for i in range(len(nus)) ] # using lihua's note, empirical risk at each lambda (adjusted)
    alpha_adjusted = alpha/(1+alpha)
    p_values = np.array([hb_p_value(r_hat,n,alpha_adjusted) for r_hat in r_hats])
    p_values = np.nan_to_num(p_values, nan=1.0)
    return bonferroni_search(p_values,delta,downsample_factor)

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

