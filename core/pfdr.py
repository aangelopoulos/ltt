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
from core.uniform_concentration import required_fdp
from core.concentration import * 
CACHE = str(Path(__file__).parent.absolute()) + '/.cache/'

"""
    RW SPECIALIZATIONS
"""

"""
    BONFERRONI SPECIALIZATIONS 
"""

# correct_vector_i = 1(top class is correct on example i)
# score_vector_i = score of top class for example i
def pfdr_bonferroni_HB(correct_vector,score_vector,lambdas,alpha,delta):
    n = correct_vector.shape[0]
    nus = [1-correct_vector[score_vector > lam].astype(float).mean() for lam in lambdas]
    rs = [(score_vector > lam).astype(float).mean() for lam in lambdas]
    r_hats = [ (nus[i]-alpha*rs[i]+alpha)/(1+alpha) for i in range(len(nus)) ] # using lihua's note, empirical risk at each lambda (adjusted)
    alpha_adjusted = alpha/(1+alpha)
    p_values = np.array([hb_p_value(r_hat,n,alpha_adjusted) for r_hat in r_hats])
    p_values = np.nan_to_num(p_values, nan=1.0)
    return bonferroni(p_values,delta)

"""
    BONFERRONI SEARCH SPECIALIZATIONS 
"""

if __name__ == "__main__":
    n = 1000
    cutoff = 0.7
    score_vector = np.random.uniform(size=(n,)) 
    correct_vector = score_vector > cutoff
    lambdas = np.linspace(0,1,20)
    alpha = 0.1
    delta = 0.1
    R = pfdr_bonferroni_HB(correct_vector, score_vector, lambdas, alpha, delta)
    print("Hello, World!")

