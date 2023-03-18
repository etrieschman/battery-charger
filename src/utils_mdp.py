import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

from utils_battery import discretize_e_states


def get_charge(e, a, b_params):
    if a == 0:
        return 0, 0
    if a == 1:
        return np.minimum(b_params['capacity'], b_params['dur']*b_params['capacity'] - e), 1/b_params['efficiency']
    return -np.minimum(b_params['capacity'], e), b_params['efficiency']

def R(e, t, a, X, y, r_weights, b_params):
    pma = X.lmp_rt_m1_rolling.iloc[t]
    p = y[t]
    charge, eff = get_charge(e, a, b_params)        
    return ((pma - p) * (charge))*(r_weights[0]) + e*(r_weights[1]) + (1-r_weights[0]-r_weights[1])*(p*(-charge*eff))


def lookahead(e, t, a, U_theta, X, y, r_weights, b_params, gamma=0.95):
    r = R(e, t, a, X, y, r_weights, b_params)
    charge, eff = get_charge(e, a, b_params)
    svect = np.hstack([X.iloc[t].values, y[t].reshape(-1,1), np.repeat(e + charge, len(t)).reshape(-1,1)])
    return r + gamma * svect @ U_theta[e + charge]


def backup(e, t, U_theta, X, y, r_weights, b_params):
    return np.array([lookahead(e, t, a, U_theta, X, y, r_weights, b_params) for a in [0,1,2]])


def fit(e, t, U, X, y):
    svect = np.hstack([X.iloc[t].values, y[t].reshape(-1,1), np.repeat(e, len(t)).reshape(-1,1)])
    return sm.OLS(endog=U, exog=svect).fit(disp=0).params


def train_valueit_model(X, y, r_weights, b_params, S=1000, kmax=200, verbose=False, seed=32):
    st_e = discretize_e_states(b_params)
    U_theta = {}
    for e in st_e:
        U_theta[e] = np.zeros(X.shape[1]+2)
    U_theta_log = np.zeros([kmax, len(st_e)])
    U_samp = {}

    np.random.seed(seed)
    T = np.random.choice(range(len(X)), S, replace=False)
    for k in tqdm(range(kmax), disable=not verbose, desc='train'):
        for i, e in enumerate(st_e):
            # get samples
            U_samp[e] = np.max(backup(e, T, U_theta, X, y, r_weights, b_params), axis=0)
            # refit U_theta, but save new estimate so i can plot convergence
            Ut = fit(e, T, U_samp[e], X, y)
            U_theta_log[k, i] = np.abs(U_theta[e]/(Ut+0.000000001) - 1).mean()
            U_theta[e] = Ut
            
    return U_theta, U_theta_log


def test_valueit_model(X, y, ypred, U_theta, hp_weights_r, b_params, verbose=False):
    N = len(X)
    a = np.zeros(N)
    c = np.zeros(N)
    e = np.zeros(N)
    revenue = np.zeros(N)

    for t in tqdm(range(N-1), disable=not verbose, desc='test'):
        # get policy
        a[t] = np.argmax(backup(e[t], [t], U_theta, X, ypred, hp_weights_r, b_params), axis=0)
        # save outcomes
        c[t], eff = get_charge(e[t], a[t], b_params)
        e[t+1] = e[t] + c[t]
        revenue[t] = y[t] * (-c[t]*eff)

    return np.array([a, c, e]), revenue