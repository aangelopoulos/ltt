import os, sys
# import from parent of absolute path of this file
curr_dir = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
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

# Load cached data
data = np.load(curr_dir + '/notebook_data/coco-tresnetxl.npz')
example_paths = os.listdir(curr_dir + '/notebook_data/examples')

sgmd = data['sgmd'] # sigmoid scores
labels = data['labels']
example_indexes = data['example_indexes']

# Problem setup
ns = [50,100,250,500,1000,2000,4000] # number of calibration points
alpha = 0.2 # 1-alpha is the desired false discovery rate
delta = 0.1 # delta is the failure rate
lambdas = np.linspace(0.1,0.9,5000)
num_trials = 10

def false_discovery_rate(prediction_set, gt_labels):
    numerator = (prediction_set * gt_labels).sum(axis=1)
    denominator = prediction_set.sum(axis=1)
    denominator[denominator == 0] = 1
    return 1-(numerator/denominator)

def true_positive_rate(prediction_set, gt_labels):
    numerator = (prediction_set * gt_labels).sum(axis=1)
    denominator = gt_labels.sum(axis=1)
    denominator[denominator == 0] = 1
    return numerator/denominator

df = []
for trial in range(num_trials):
    for n in ns:
# Split the softmax scores into calibration and validation sets (save the shuffling)
        idx = np.array([1] * n + [0] * (sgmd.shape[0]-n)) > 0
        np.random.shuffle(idx)
        cal_sgmd, val_sgmd = sgmd[idx,:], sgmd[~idx,:]
        cal_labels, val_labels = labels[idx], labels[~idx]


# Scan to choose lambda hat (Bentkus)
        for lhat_bentkus in np.flip(lambdas):
            fdrs = false_discovery_rate(cal_sgmd >= (lhat_bentkus - 1/len(lambdas)), cal_labels)
            if hb_p_value(fdrs.mean(), n, alpha) > delta: break
# Deploy procedure on test data
        prediction_sets_bentkus = val_sgmd >= lhat_bentkus

# Scan to choose lambda hat (Betting)
        for lhat_betting in np.flip(lambdas):
            fdrs = false_discovery_rate(cal_sgmd >= (lhat_betting - 1/len(lambdas)), cal_labels)
            if wsr_p_value(fdrs, alpha, delta=delta) > delta: break
# Deploy procedure on test data
        prediction_sets_betting = val_sgmd >= lhat_betting

# Scan to choose lambda hat (CLT)
        for lhat_clt in np.flip(lambdas):
            fdrs = false_discovery_rate(cal_sgmd >= (lhat_clt - 1/len(lambdas)), cal_labels)
            if clt_p_value(fdrs.mean(),  fdrs.std(), n, alpha) > delta: break

# Deploy procedure on test data
        prediction_sets_clt = val_sgmd >= lhat_clt

        df = df + [
            pd.DataFrame([{
                "FDR" : false_discovery_rate(prediction_sets_bentkus, val_labels).mean(),
                "TPR" : true_positive_rate(prediction_sets_bentkus, val_labels).mean(),
                "lhat" : lhat_bentkus,
                "n" : n,
                "bound" : "Bentkus",
                "trial" : trial,
                }])
            ]

        df = df + [
            pd.DataFrame([{
                "FDR" : false_discovery_rate(prediction_sets_betting, val_labels).mean(),
                "TPR" : true_positive_rate(prediction_sets_betting, val_labels).mean(),
                "lhat" : lhat_betting,
                "n" : n,
                "bound" : "Betting",
                "trial" : trial,
                }])
            ]

        df = df + [
            pd.DataFrame([{
                "FDR" : false_discovery_rate(prediction_sets_clt, val_labels).mean(),
                "TPR" : true_positive_rate(prediction_sets_clt, val_labels).mean(),
                "lhat" : lhat_clt,
                "n" : n,
                "bound" : "CLT",
                "trial" : trial,
                }])
            ]

df = pd.concat(df)

# Pivot the table to get the desired format
pivot_df = df.pivot_table(index='n', columns='bound', values=['FDR', 'TPR', 'lhat'])

# Output the LaTeX code
latex_table = pivot_df.to_latex(multicolumn=True, multicolumn_format='c', escape=False, float_format="%.3f")
print(latex_table)
