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
data = np.load(curr_dir + '/notebook_data/imagenet-resnet152.npz')
print(data.files)

smx = data['smx']
labels = data['labels']
corrects = (smx.argmax(axis=1) == labels).astype(int)

# Problem setup
ns = [100, 500, 1000,2000,3000,3946] # number of calibration points
alpha = 0.15 # 1-alpha is the desired accuracy
delta = 0.2 # delta is the failure rate
lambdas = np.linspace(0,1,1000)
num_trials = 100
n_min = 20


df = []
for n in ns:
    # Fix randomness
    np.random.seed(0)
    
    for trial in tqdm(range(num_trials)):
# Split the softmax scores into calibration and validation sets (save the shuffling)
        idx = np.array([1] * n + [0] * (smx.shape[0]-n)) > 0
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx], smx[~idx]
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_corrects, val_corrects = corrects[idx], corrects[~idx]
        cal_scores, val_scores = cal_smx.max(axis=1), val_smx.max(axis=1)


# Scan to choose lambda hat (Bentkus)
        first_bentkus = True
        for lhat_bentkus in reversed(lambdas):
            _idx = cal_scores >= (lhat_bentkus - 1/len(lambdas))
            _n = _idx.sum()
            _cal_corrects = cal_corrects[_idx]
            _hb_pval = hb_p_value(1-_cal_corrects.mean(), _n, alpha)
            if (_n > n_min) and (_hb_pval > delta): 
                if first_bentkus:
                    lhat_bentkus = lambdas[-1]
                break
            elif (_n > n_min) and (_hb_pval <= delta):
                first_bentkus = False

# Scan to choose lambda hat (Betting)
        first_betting = True
        for lhat_betting in reversed(lambdas):
            _idx = cal_scores >= (lhat_betting - 1/len(lambdas))
            _n = _idx.sum()
            _cal_corrects = cal_corrects[_idx]
            if (_n > 1):
                _wsr_pval = wsr_p_value(1-_cal_corrects, alpha, delta=delta)
            else:
                continue
            if (_n > n_min) and (_wsr_pval > delta): 
                if first_betting:
                    lhat_betting = lambdas[-1]
                break
            elif (_n > n_min) and (_wsr_pval <= delta):
                first_betting = False

# Scan to choose lambda hat (CLT)
        first_clt = True
        for lhat_clt in reversed(lambdas):
            _idx = cal_scores >= (lhat_clt - 1/len(lambdas))
            _n = _idx.sum()
            _cal_corrects = cal_corrects[_idx]
            _clt_pval = clt_p_value(1-_cal_corrects.mean(),  _cal_corrects.std(), _n, alpha)
            if (_n > n_min) and (_clt_pval > delta): 
                if first_clt:
                    lhat_clt = lambdas[-1]

                break
            elif (_n > n_min) and (_clt_pval <= delta):
                first_clt = False

        df = df + [
            pd.DataFrame([{
                "accuracy" : np.nan_to_num(val_corrects[val_scores >= lhat_bentkus].mean()),
                "TPR" : np.nan_to_num((val_scores >= lhat_bentkus)[val_corrects].mean()),
                "lhat" : lhat_bentkus,
                "n" : n,
                "bound" : "Bentkus",
                "trial" : trial,
                }])
            ]

        df = df + [
            pd.DataFrame([{
                "accuracy" : np.nan_to_num(val_corrects[val_scores >= lhat_betting].mean()),
                "TPR" : np.nan_to_num((val_scores >= lhat_betting)[val_corrects].mean()),
                "lhat" : lhat_betting,
                "n" : n,
                "bound" : "Betting",
                "trial" : trial,
                }])
            ]

        df = df + [
            pd.DataFrame([{
                "accuracy" : np.nan_to_num(val_corrects[val_scores >= lhat_clt].mean()),
                "TPR" : np.nan_to_num((val_scores >= lhat_clt)[val_corrects].mean()),
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
quantile_df = df.pivot_table(index='n', columns='bound', values=['accuracy', 'TPR', 'lhat'], aggfunc=[quantile])
quantile_df.columns = quantile_df.columns.droplevel(0)
mean_df = df.pivot_table(index='n', columns='bound', values=['accuracy', 'TPR', 'lhat'], aggfunc='mean')
# Drop MSE (+) from mean table and replace with MSE (+) from quantile table
mean_df = mean_df.drop(columns='accuracy', level=0)
mean_df = pd.concat([ quantile_df.drop(columns=['TPR', 'lhat'], level=0), mean_df], axis=1)
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