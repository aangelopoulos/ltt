import os, itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest 
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr
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

def optimize_params_GBR(X_train, X_val, y_train, y_val):
    filename = './.cache/GBR_optim.pkl'
    try:
        optim_df = pd.read_pickle(filename)
    except:
        lrs = list(np.logspace(-2,0,3))
        n_ests = [50,100,500]
        subsamples = [0.5,1] 
        max_depths = [10,25,] 
        optim_df = pd.DataFrame(columns = ['lr','n_estimators', 'subsample', 'max_depth', 'mse'])
        for lr, n_estimators, subsample, max_depth in tqdm(itertools.product(lrs, n_ests, subsamples, max_depths)):
            regr = GradientBoostingRegressor(random_state=0, learning_rate=lr, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
            regr.fit(X_train, y_train)
            pred = regr.predict(X_val)
            mse = ((pred - y_val)**2).mean()
            optim_dict = { 'lr' : lr,'n_estimators' : n_estimators, 'subsample' : subsample, 'max_depth' : max_depth, 'mse' : mse }
            optim_df = optim_df.append(optim_dict, ignore_index=True)
            tqdm.write(str(optim_dict))
        os.makedirs('./.cache/', exist_ok=True)
        optim_df.to_pickle(filename)
    idx = np.argmin(optim_df['mse'])
    optim_df = optim_df.loc[idx]
    print(f"Optimal Params: {optim_df}")
    # GBR
    regr = GradientBoostingRegressor(random_state=0, learning_rate=optim_df['lr'], n_estimators=int(optim_df['n_estimators']), subsample=optim_df['subsample'], max_depth=int(optim_df['max_depth']))
    regr.fit(X_train, y_train)
    pred = regr.predict(X_val)
    # Anomaly detector
    clf = LocalOutlierFactor(novelty=True, n_neighbors=200).fit(X_train)
    anomalyScores = -clf.score_samples(X_val) # larger is more anomalous
    residuals = (pred - y_val)
    pdb.set_trace()
    mse = (residuals**2).mean()
    return regr 


if __name__ == "__main__":
    X, y = get_data()
    X_train, X_val, y_train, y_val = process_data(*shuffle_split(X,y))
    regr = optimize_params_GBR(X_train, X_val, y_train, y_val)
    pdb.set_trace()
    print(y_val)

