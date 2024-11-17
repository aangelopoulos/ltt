import os, sys
# import from parent of absolute path of this file
curr_dir = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(dir_path)
sys.path.append(dir_path)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.optimize import brentq
from scipy.stats import norm, binom
from core.bounds import hb_p_value, wsr_p_value, clt_p_value
from confseq import betting
import pdb
from tqdm import tqdm

# Load cached data
X_val = np.load(curr_dir + '/.cache/X_val.npy')
y_val = np.load(curr_dir + '/.cache/y_val.npy')
mean_val = np.load(curr_dir + '/.cache/mean_val.npy')
upper_val = np.load(curr_dir + '/.cache/upper_val.npy')
lower_val = np.load(curr_dir + '/.cache/lower_val.npy')
error_scores_val = np.load(curr_dir + '/.cache/error_scores_val.npy')
squared_error_val = (y_val-mean_val)**2
squared_error_val /= squared_error_val.max()

# Problem setup
ns = [500, 1000,2000,3000,3946] # number of calibration points
alpha = 0.15 # 1-alpha is the desired false discovery rate
delta = 0.2 # delta is the failure rate
lambdas = np.linspace(np.quantile(error_scores_val, 0.01),np.quantile(error_scores_val,0.99),1000)
num_trials = 20
n_min = 100


df = []
for n in ns:
    # Fix randomness
    np.random.seed(0)
    
    for trial in tqdm(range(num_trials)):
# Split the softmax scores into calibration and validation sets (save the shuffling)
        idx = np.array([1] * n + [0] * (error_scores_val.shape[0]-n)) > 0
        np.random.shuffle(idx)
        cal_error_scores, val_error_scores = error_scores_val[idx], error_scores_val[~idx]
        cal_squared_error, val_squared_error = squared_error_val[idx], squared_error_val[~idx]


# Scan to choose lambda hat (Bentkus)
        first_bentkus = True
        for lhat_bentkus in lambdas:
            _idx = cal_error_scores <= (lhat_bentkus + 1/len(lambdas))
            _n = _idx.sum()
            _cal_squared_error = cal_squared_error[_idx]
            _hb_pval = hb_p_value(_cal_squared_error.mean(), _n, alpha)
            if (_n > n_min) and (_hb_pval > delta): 
                if first_bentkus:
                    lhat_bentkus = lambdas[0]
                break
            elif (_n > n_min) and (_hb_pval <= delta):
                first_bentkus = False

# Scan to choose lambda hat (Betting)
        first_betting = True
        for lhat_betting in lambdas:
            _idx = cal_error_scores <= (lhat_betting + 1/len(lambdas))
            _n = _idx.sum()
            _cal_squared_error = cal_squared_error[_idx]
            if (_n > 1):
                _wsr_pval = wsr_p_value(_cal_squared_error, alpha, delta=delta)
            else:
                continue
            if (_n > n_min) and (_wsr_pval > delta): 
                if first_betting:
                    lhat_betting = lambdas[0]
                break
            elif (_n > n_min) and (_wsr_pval <= delta):
                first_betting = False

# Scan to choose lambda hat (CLT)
        first_clt = True
        for lhat_clt in lambdas:
            _idx = cal_error_scores <= (lhat_clt + 1/len(lambdas))
            _idx_val = val_error_scores <= (lhat_clt + 1/len(lambdas))
            _n = _idx.sum()
            _cal_squared_error = cal_squared_error[_idx]
            _val_squared_error = val_squared_error[_idx_val]
            _clt_pval = clt_p_value(_cal_squared_error.mean(),  _cal_squared_error.std(), _n, alpha)
            if (_n > n_min) and (_clt_pval > delta): 
                if first_clt:
                    lhat_clt = lambdas[0]

                break
            elif (_n > n_min) and (_clt_pval <= delta):
                first_clt = False

        df = df + [
            pd.DataFrame([{
                "MSE (+)" : np.nan_to_num(val_squared_error[val_error_scores <= lhat_bentkus].mean()),
                "MSE (-)" : np.nan_to_num(val_squared_error[val_error_scores > lhat_bentkus].mean()),
                "frac acc pred" : np.nan_to_num((val_error_scores <= lhat_bentkus)[val_squared_error <= alpha].mean()),
                "lhat" : lhat_bentkus,
                "n" : n,
                "bound" : "Bentkus",
                "trial" : trial,
                }])
            ]

        df = df + [
            pd.DataFrame([{
                "MSE (+)" : np.nan_to_num(val_squared_error[val_error_scores <= lhat_betting].mean()),
                "MSE (-)" : np.nan_to_num(val_squared_error[val_error_scores > lhat_betting].mean()),
                "frac acc pred" : np.nan_to_num((val_error_scores <= lhat_betting)[val_squared_error <= alpha].mean()),
                "lhat" : lhat_betting,
                "n" : n,
                "bound" : "Betting",
                "trial" : trial,
                }])
            ]

        df = df + [
            pd.DataFrame([{
                "MSE (+)" : np.nan_to_num(val_squared_error[val_error_scores <= lhat_clt].mean()),
                "MSE (-)" : np.nan_to_num(val_squared_error[val_error_scores > lhat_clt].mean()),
                "frac acc pred" : np.nan_to_num((val_error_scores <= lhat_clt)[val_squared_error <= alpha].mean()),
                "lhat" : lhat_clt,
                "n" : n,
                "bound" : "CLT",
                "trial" : trial,
                }])
            ]

df = pd.concat(df)

def quantile(s:pd.Series):
    return s.quantile(1-delta)

# Pivot the table to get the desired format
quantile_df = df.pivot_table(index='n', columns='bound', values=['MSE (+)', 'frac acc pred', 'lhat'], aggfunc=[quantile])
quantile_df.columns = quantile_df.columns.droplevel(0)
mean_df = df.pivot_table(index='n', columns='bound', values=['MSE (+)', 'frac acc pred', 'lhat'], aggfunc='mean')
# Drop MSE (+) from mean table and replace with MSE (+) from quantile table
mean_df = mean_df.drop(columns='MSE (+)', level=0)
mean_df = pd.concat([ quantile_df.drop(columns=['frac acc pred', 'lhat'], level=0), mean_df], axis=1)
# Output the LaTeX code

def formatFloat(val):
  ret = "%.3f" % val
  if ret.startswith("0."):
    return ret[1:]
  if ret.startswith("-0."):
    return "-" + ret[2:]
  return ret

latex_table = mean_df.to_latex(multicolumn=True, multicolumn_format='c', escape=False, float_format=formatFloat)
print(latex_table)