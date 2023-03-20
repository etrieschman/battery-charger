import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.model_selection import LeavePGroupsOut

from utils_mdp_m import train_valueit_model, test_valueit_model
from utils_mdp_mf import train_Q, validate_Q, Q_ITERS, Q_CHUNKSIZE


def train_test_split(df, X_cols, y_col, group_cols=None, yr_val=2021):
    # split data
    idx_val = (df.year == yr_val).values
    X, y = df[X_cols], df[y_col].values
    X_val, y_val = X.loc[idx_val], y[idx_val]
    X_tt, y_tt = X.loc[~idx_val], y[~idx_val]
    if group_cols is None:
        return X_tt, y_tt, X_val, y_val
    
    groups_tt = df.loc[~idx_val, group_cols].astype(str).apply(''.join, axis=1).values
    return X_tt, y_tt, groups_tt, X_val, y_val


def crossval_Q(states, y, groups, b_params, hp_Q, hp_R, verbose=False, q_iters=Q_ITERS, q_chunksize=Q_CHUNKSIZE, n_groups=1):
    s_e, s_t, s_h, s_dow, s_rt = states
    cvsum = {'cumrev':[], 'cumrew':[], 'mean_storage':[], 'downtime':[]}
    
    tt_lpgo_splitter = LeavePGroupsOut(n_groups=n_groups)
    splits = list(tt_lpgo_splitter.split(y, groups=groups))
    for train_idx, test_idx in tqdm(splits, disable=not verbose):

        # get states
        train_states = s_e, s_t[train_idx], s_h[train_idx], s_dow[train_idx], s_rt[train_idx]
        test_states = s_e, s_t[test_idx], s_h[test_idx], s_dow[test_idx], s_rt[test_idx]

        # train/test
        Q, __, __ = train_Q(train_states, y[train_idx], b_params, hp_Q, hp_R, q_iters, q_chunksize, False)
        ace, revrew = validate_Q(Q, test_states, y[test_idx], b_params, hp_R, False)

        # record
        cvsum['cumrev'] += [revrew[0].sum()]
        cvsum['cumrew'] += [revrew[1].sum()]
        cvsum['mean_storage'] += [ace[2].mean()]
        cvsum['downtime'] += [(ace[0] == 0).mean()]

    return cvsum


# OLD


def crossval_VI(X, Xvi_cols, y, groups, hp_weights, b_params, R, S=None, kmax=200, yr_val=2022, desc='cv'):
    # split into train and test data
    tt_lpgo_splitter = LeavePGroupsOut(n_groups=1)
    splits = list(tt_lpgo_splitter.split(X, y, groups))
    
    # create log dictionaries
    # log = {'Utheta':[], 'Utheta_log':[], 'ace':[], 'revenue':[]}
    cvsum = {'cumrev':[], 'mean_storage':[], 'downtime':[]}

    # cross validate
    for train_idx, test_idx in tqdm(splits, desc=desc):
        Utheta, __ = train_valueit_model(
            X[Xvi_cols].iloc[train_idx], y[train_idx], hp_weights, b_params, R, S, kmax, verbose=False)

        # test with price forecast
        ymodel = sm.OLS(endog=y[train_idx], exog=X.iloc[train_idx]).fit(disp=0)
        yhat_test = ymodel.predict(X.iloc[test_idx]).values
        ace, revenue = test_valueit_model(
            X[Xvi_cols].iloc[test_idx], y[test_idx], yhat_test, Utheta, hp_weights, b_params, R, verbose=False)

        # record
        cvsum['cumrev'] += [revenue.sum()]
        cvsum['mean_storage'] += [ace[2].mean()]
        cvsum['downtime'] += [(ace[0] == 0).mean()]

    cvsummean = {'cumrev_range': np.array(cvsum['cumrev']).max() - np.array(cvsum['cumrev']).min(),
                 'cumrev_gt0': (np.array(cvsum['cumrev']) > 0).sum()}
    for k, v in cvsum.items():
        cvsummean[k] = np.array(v).mean()
        
    return cvsum, cvsummean
