import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


def discretize_e_states(b_params):
    return np.linspace(0, b_params['dur']*b_params['capacity'], b_params['dur']*4+1)

def get_charge(e, a, b_params, return_eff=False):
    if a == 0:
        out = [0, 0]
    elif a == 1:
        out = [np.minimum(b_params['capacity'], b_params['dur']*b_params['capacity'] - e), 1/b_params['efficiency']]
    else:
        out = [-np.minimum(b_params['capacity'], e), b_params['efficiency']]

    if return_eff:
        return out
    return out[0]

def R2(e, t, a, X, y, hp, b_params): # original version of reward function
    pma = X.lmp_rt_m1_rolling.iloc[t]
    p = y[t]
    charge, eff = get_charge(e, a, b_params, True) 
    
    r = (1-hp[0]*hp[1]) * ((pma - p)*(charge))
    r += (1-hp[0]*(1-hp[1])) * (e + charge)
    r += hp[0] * (p*(-charge*eff))
    return r

def R(e, t, a, X, y, hp, b_params):
    pma = X.lmp_rt_m1_rolling.iloc[t]
    p = y[t]
    charge, eff = get_charge(e, a, b_params, True) 
    
    r = (1-hp[0]) * ((pma - p)*(charge))
    r += hp[1] * (b_params['dur']*b_params['capacity']/(e + b_params['capacity']/2))*(a==1)
    r += hp[0] * (p*(-charge*eff))
    return r


def lookahead(e, t, a, U_theta, X, y, hp, b_params, R=R, gamma=0.95):
    r = R(e, t, a, X, y, hp, b_params)
    charge = get_charge(e, a, b_params)
    # svect = np.hstack([X.iloc[t].values, y[t].reshape(-1,1), np.repeat(e + charge, len(t)).reshape(-1,1)])
    svect = (np.hstack([X.iloc[t].values, y[t].reshape(-1,1)]))
    return r + gamma * svect @ U_theta[e + charge]

# def lookaheadL(e, t, a, U_theta, X, y, hp, b_params, R=R, gamma=0.95):
#     r = R(e, t, a, X, y, hp, b_params)
#     charge = get_charge(e, a, b_params)
#     # svect = np.hstack([X.iloc[t].values, y[t].reshape(-1,1), np.repeat(e + charge, len(t)).reshape(-1,1)])
#     svect = X.iloc[t].copy()
#     svect[f'e{int(e)}'] = 1
#     for col in [col for col in X.columns if not (col.startswith('e_') | col.find('_x_e'))]:
#         svect[f'{col}_x_e{int(e)}'] = svect[f'{col}']
#     svect['y'] = y[t]
#     return r + gamma * svect @ U_theta


def backup(e, t, U_theta, X, y, hp, b_params, R=R):
    return np.array([lookahead(e, t, a, U_theta, X, y, hp, b_params, R) for a in [0,1,2]])

# def backupL(e, t, U_theta, X, y, hp, b_params, R=R):
#     return np.array([lookaheadL(e, t, a, U_theta, X, y, hp, b_params, R) for a in [0,1,2]])


def fit(e, t, U, X, y):
    # svect = np.hstack([X.iloc[t].values, y[t].reshape(-1,1), np.repeat(e, len(t)).reshape(-1,1)])
    svect = X.iloc[t].copy()
    svect['y'] = y[t]
    return sm.OLS(endog=U, exog=svect).fit(disp=0).params

# def fitL(e, t, U, X, y):
#     # svect = np.hstack([X.iloc[t].values, y[t].reshape(-1,1), np.repeat(e, len(t)).reshape(-1,1)])
#     svect = X.iloc[t].copy()
#     svect[f'e{int(e)}'] = 1
#     for col in [col for col in X.columns if not (col.startswith('e_') | col.find('_x_e'))]:
#         svect[f'{col}_x_e{int(e)}'] = svect[f'{col}']
#     svect['y'] = y[t]
#     return sm.OLS(endog=U, exog=svect).fit(disp=0).params


def train_valueit_model(X, y, hp, b_params, R, S=None, kmax=200, verbose=False, seed=32):
    st_e = discretize_e_states(b_params)
    U_theta = {}
    for e in st_e:
        U_theta[e] = np.zeros(X.shape[1]+1)
    U_theta_log = np.zeros([kmax, len(st_e)])
    U_samp = {}

    np.random.seed(seed)
    if S is None:
        T = np.arange(len(X)) # train on whole dataset because why not?
    else:
        T = np.random.choice(range(len(X)), S, replace=False)
    
    for k in tqdm(range(kmax), disable=not verbose, desc='train'):
        for i, e in enumerate(st_e):
            # get samples
            U_samp[e] = np.max(backup(e, T, U_theta, X, y, hp, b_params, R), axis=0)
            # refit U_theta, but save new estimate so i can plot convergence
            Ut = fit(e, T, U_samp[e], X, y)
            U_theta_log[k, i] = np.abs(U_theta[e]/(Ut+0.000000001) - 1).mean()
            U_theta[e] = Ut
            
    return U_theta, U_theta_log

# def train_valueit_modelL(X, y, hp, b_params, R, S=None, kmax=200, verbose=False, seed=32):
#     st_e = discretize_e_states(b_params)
    
#     # make array for fitting
#     colnames = []
#     for c in st_e.astype(int):
#         colnames += [f'e{c}']
#     for col in [col for col in X.columns if not col.startswith('e_')]:
#         colnames += [f'{col}_x_e{c}']
#     newcols = pd.DataFrame(np.zeros([len(X), len(colnames)]), columns=colnames)
#     Xvi = pd.concat([X.reset_index(drop=True), newcols], axis=1)
#     U_theta = np.zeros(Xvi.shape[1] + 1)
#     U_theta_log = np.zeros([kmax])

#     np.random.seed(seed)
#     if S is None:
#         T = np.arange(len(X)) # train on whole dataset because why not?
#     else:
#         T = np.random.choice(range(len(X)), S, replace=False)
    
#     for k in tqdm(range(kmax), disable=not verbose, desc='train'):
#         # get samples
#         U_samp = np.max(backupL(st_e, T, U_theta, Xvi, y, hp, b_params, R), axis=0)
#         # refit U_theta, but save new estimate so i can plot convergence
#         Ut = fitL(st_e, T, U_samp, Xvi, y)
#         U_theta_log[k] = np.abs(U_theta/(Ut+0.000000001) - 1).mean()
#         U_theta = Ut
            
#     return U_theta, U_theta_log


def test_valueit_model(X, y, ypred, U_theta, hp, b_params, R=R, verbose=False):
    N = len(X)
    a = np.zeros(N)
    c = np.zeros(N)
    e = np.zeros(N)
    revenue = np.zeros(N)

    for t in tqdm(range(N-1), disable=not verbose, desc='test'):
        # get policy
        a[t] = np.argmax(backup(e[t], [t], U_theta, X, ypred, hp, b_params, R), axis=0)
        # save outcomes
        c[t], eff = get_charge(e[t], a[t], b_params, True)
        e[t+1] = e[t] + c[t]
        revenue[t] = y[t] * (-c[t]*eff)

    return np.array([a, c, e]), revenue