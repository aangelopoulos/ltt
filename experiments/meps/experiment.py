import os, sys, inspect, itertools
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest 
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr
from core.concentration import *
from tqdm import tqdm
import pdb

def get_data():
    #df = pd.concat( [ pd.read_csv('./data/meps_19_reg.csv'), pd.read_csv('./data/meps_20_reg.csv'), pd.read_csv('./data/meps_21_reg.csv') ] )
    df = pd.read_csv('./data/meps_19_reg.csv')
    response_name = "UTILIZATION_reg"
    col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
               'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
               'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
               'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
               'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
               'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
               'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
               'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
               'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
               'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
               'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
               'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
               'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
               'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
               'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
               'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
               'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
               'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
               'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
               'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
               'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
               'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
               'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
               'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
               'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
               'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
               'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

    y = df[response_name].values.astype(np.float32)
    X = df[col_names].values.astype(np.float32)

    return X, y

def shuffle_split(X,y):
    n_full = X.shape[0]
    perm = np.random.permutation(n_full)
    X = X[perm]
    y = y[perm]
    n = n_full//2
    return X[:n], X[n:], y[:n], y[n:]

def process_data(X_train, X_val, y_train, y_val):
	# zero mean and unit variance scaling 
	scalerX = StandardScaler()
	scalerX = scalerX.fit(X_train)
	X_train = scalerX.transform(X_train)
	X_val = scalerX.transform(X_val)

	# scale the response as it is highly skewed
	y_train = np.log(1.0 + y_train)
	y_val = np.log(1.0 + y_val)
	# reshape the response
	y_train = np.squeeze(np.asarray(y_train))
	y_val = np.squeeze(np.asarray(y_val))
	return X_train, X_val, y_train, y_val

def optimize_params_GBR(X_train, X_val, y_train, y_val, alpha=0.1):
    filename = './.cache/GBR_optim.pkl'
    try:
        optim_df = pd.read_pickle(filename)
    except:
        lrs = [0.01,]
        n_ests = [100,]
        subsamples = [1,]
        max_depths = [10,25] 
        optim_df = pd.DataFrame(columns = ['lr','n_estimators', 'subsample', 'max_depth', 'cvg'])
        for lr, n_estimators, subsample, max_depth in tqdm(itertools.product(lrs, n_ests, subsamples, max_depths)):
            mean = GradientBoostingRegressor(random_state=0, learning_rate=lr, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
            upper = GradientBoostingRegressor(random_state=0, learning_rate=lr, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth, alpha=1-alpha/2, loss='quantile')
            lower = GradientBoostingRegressor(random_state=0, learning_rate=lr, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth, alpha=alpha/2, loss='quantile')
            mean.fit(X_train, y_train)
            upper.fit(X_train, y_train)
            lower.fit(X_train, y_train)
            pred_mean = mean.predict(X_val) 
            pred_upper = upper.predict(X_val)
            pred_lower = lower.predict(X_val)
            pred_upper = np.maximum(pred_upper, pred_lower + 1e-6)
            mse = ( (y_val - pred_mean)**2 ).mean()
            cvg = ( (y_val <= pred_upper) & (y_val >= pred_lower) ).mean()
            optim_dict = { 'lr' : lr,'n_estimators' : n_estimators, 'subsample' : subsample, 'max_depth' : max_depth, 'mse' : mse, 'cvg' : cvg }
            optim_df = optim_df.append(optim_dict, ignore_index=True)
            tqdm.write(str(optim_dict))
        os.makedirs('./.cache/', exist_ok=True)
        optim_df.to_pickle(filename)
    idx_quantiles = np.argmin(np.abs(optim_df['cvg']-(1-alpha)))
    idx_mean = np.argmin(optim_df['mse'])
    optim_df_quantiles = optim_df.loc[idx_quantiles]
    optim_df_mean = optim_df.loc[idx_mean]
    # GBR
    mean = GradientBoostingRegressor(random_state=0, learning_rate=optim_df_mean['lr'], n_estimators=int(optim_df_mean['n_estimators']), subsample=optim_df_mean['subsample'], max_depth=int(optim_df_mean['max_depth']), alpha=1-alpha/2, loss='quantile')
    upper = GradientBoostingRegressor(random_state=0, learning_rate=optim_df_quantiles['lr'], n_estimators=int(optim_df_quantiles['n_estimators']), subsample=optim_df_quantiles['subsample'], max_depth=int(optim_df_quantiles['max_depth']), alpha=1-alpha/2, loss='quantile')
    lower = GradientBoostingRegressor(random_state=0, learning_rate=optim_df_quantiles['lr'], n_estimators=int(optim_df_quantiles['n_estimators']), subsample=optim_df_quantiles['subsample'], max_depth=int(optim_df_quantiles['max_depth']), alpha=alpha/2, loss='quantile')
    mean.fit(X_train, y_train)
    upper.fit(X_train, y_train)
    lower.fit(X_train, y_train)
    pred_mean_train = mean.predict(X_train)
    pred_mean_val = mean.predict(X_val)
    pred_upper_val = upper.predict(X_val)
    pred_lower_val = lower.predict(X_val)
    residual_magnitudes_train = np.abs( pred_mean_train - y_train )
    residual_magnitudes_val = np.abs( pred_mean_val - y_val )
    coverage_indicators_val = (pred_lower_val <= y_val) & (pred_upper_val >= y_val)
    # Anomaly detector
    error_predictor = GradientBoostingRegressor(random_state=0, learning_rate=optim_df_mean['lr'], n_estimators=int(optim_df_mean['n_estimators']), subsample=optim_df_mean['subsample'], max_depth=int(optim_df_mean['max_depth']))
    error_predictor.fit(X_train, residual_magnitudes_train)
    error_scores_val = error_predictor.predict(X_val)
    return pred_mean_val, pred_upper_val, pred_lower_val, error_scores_val

def get_model_outputs():
    try:
        X_val = np.load('./.cache/X_val.npy')
        y_val = np.load('./.cache/y_val.npy')
        mean_val = np.load('./.cache/mean_val.npy')
        upper_val = np.load('./.cache/upper_val.npy')
        lower_val = np.load('./.cache/lower_val.npy')
        error_scores_val = np.load('./.cache/error_scores_val.npy')
    except:
        X, y = get_data()
        X_train, X_val, y_train, y_val = process_data(*shuffle_split(X,y))
        mean_val, upper_val, lower_val, error_scores_val = optimize_params_GBR(X_train, X_val, y_train, y_val)
        os.makedirs('./.cache/', exist_ok=True)
        np.save('./.cache/X_val.npy', X_val)
        np.save('./.cache/y_val.npy', y_val)
        np.save('./.cache/mean_val.npy', mean_val)
        np.save('./.cache/upper_val.npy', upper_val)
        np.save('./.cache/lower_val.npy', lower_val)
        np.save('./.cache/error_scores_val.npy', error_scores_val)
    return X_val, y_val, mean_val, upper_val, lower_val, error_scores_val 

def get_loss_table(alpha):
    try:
        # encode loss (MSE) as 0th channel and abstentions as 1st channel.
        loss_table = np.load(f'./.cache/{alpha}_loss_table.npy')
    except:
        X_val, y_val, mean_val, upper_val, lower_val, error_scores_val = get_model_outputs()
        lambdas = np.array( [ np.quantile(error_scores_val, q) for q in np.linspace(0,1,10) ] )
        loss_table = np.zeros( (y_val.shape[0], 2, lambdas.shape[0]) )
        squared_errors = (mean_val - y_val)**2
        for i in range(lambdas.shape[0]):
            bool_predict = error_scores_val <= lambdas[i]
            # encode loss (MSE) as 0th channel and abstentions as 1st channel.
            loss_table[:,0,i] = (squared_errors * bool_predict) - alpha*bool_predict + alpha
            loss_table[:,1,i] = (~bool_predict).astype(float)
        np.save('./.cache/{alpha}_loss_table.npy', loss_table)
    return loss_table

def ltt_calibrate_evaluate(loss_table, alpha, delta):
    # encode loss (MSE) as 0th channel and abstentions as 1st channel.
    np.random.shuffle(loss_table)
    mse_table, abstention_table = loss_table[:,0,:], loss_table[:,1,:]
    mse_max = mse_table.max()
    mse_table = mse_table/mse_max # scale all to 0-1
    n = mse_table.shape[0]//2
    cal_table, val_table = mse_table[:n], mse_table[n:]
    cal_risks, val_risks = cal_table.mean(axis=0), val_table.mean(axis=0)
    val_abstentions = abstention_table[n:].mean(axis=0)
    def non_vectorized_hbpv_alpha(r_hat):
        return hb_p_value(r_hat, n, alpha)
    hbpv = np.vectorize(non_vectorized_hbpv_alpha)
    pvals = hbpv(cal_risks)
    valid = bonferroni_search(pvals, delta, 1)
    ihat = valid[np.argmax( cal_risks[valid] )] # Get the minimum risk index
    risk_est = val_risks[ihat]
    pdb.set_trace()
    abstention_est = val_abstentions[ihat] 

if __name__ == "__main__":
    alpha = 0.1 
    delta = 0.1
    loss_table = get_loss_table(alpha)
    ltt_calibrate_evaluate(loss_table, alpha, delta)
