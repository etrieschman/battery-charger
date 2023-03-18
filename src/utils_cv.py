import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.model_selection import LeavePGroupsOut

from utils_mdp import train_valueit_model, test_valueit_model


def train_test_split(df, X_cols, y_col, group_cols, yr_val=2022):
    # split data
    idx_val = (df.year == yr_val).values
    X, y = df[X_cols], df[y_col].values
    X_val, y_val = X.loc[idx_val], y[idx_val]
    X_tt, y_tt = X.loc[~idx_val], y[~idx_val]
    groups_tt = df.loc[~idx_val, group_cols].astype(str).apply(''.join, axis=1).values
    
    return X_tt, y_tt, groups_tt, X_val, y_val


def crossval_model(X, y, groups, hp_weights, b_params, S=100, kmax=200, yr_val=2022, desc='cv'):
    # split into train and test data
    tt_lpgo_splitter = LeavePGroupsOut(n_groups=1)
    splits = list(tt_lpgo_splitter.split(X, y, groups))
    
    # create log dictionaries
    # log = {'Utheta':[], 'Utheta_log':[], 'ace':[], 'revenue':[]}
    cvsum = {'cumrev':[], 'mean_storage':[], 'downtime':[]}

    # cross validate
    for train_idx, test_idx in tqdm(splits, desc=desc):
        Utheta, __ = train_valueit_model(
            X.iloc[train_idx], y[train_idx], hp_weights, b_params, S, kmax, verbose=False)

        # test with price forecast
        ymodel = sm.OLS(endog=y[train_idx], exog=X.iloc[train_idx]).fit(disp=0)
        yhat_test = ymodel.predict(X.iloc[test_idx]).values
        ace, revenue = test_valueit_model(
            X.iloc[test_idx], yhat_test, Utheta, hp_weights, b_params, verbose=False)

        # record
        cvsum['cumrev'] += [revenue.sum()]
        cvsum['mean_storage'] += [ace[2].mean()]
        cvsum['downtime'] += [(ace[0] == 0).mean()]

    cvsummean = {'cumrev_range': np.array(cvsum['cumrev']).max() - np.array(cvsum['cumrev']).min(),
                 'cumrev_gt0': (np.array(cvsum['cumrev']) > 0).sum()}
    for k, v in cvsum.items():
        cvsummean[k] = np.array(v).mean()
        
    return cvsum, cvsummean
