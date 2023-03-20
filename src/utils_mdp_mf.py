import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_data import PATH_DATA

A_LOOKUP = np.array([0, 1, -1])
Q_ITERS, Q_CHUNKSIZE = 100, 10000
PATH_RESULTS = f'{PATH_DATA}results/'

def get_discrete_states(X, y, b_params, y_q=None, tcol='month', m=100):
    s_e = np.linspace(0, b_params['dur']*b_params['capacity'], b_params['dur']*4+1)
    s_h = (X.hour.values)
    s_dow = (X.dow.values.astype(int))
    s_t = (X[tcol].values.astype(int))

    if y_q is None:
        q = np.linspace(0, 1, m+1)
        y_q = np.quantile(y, q) 
    s_rt = np.clip(np.searchsorted(y_q, y), 0, m-1)
    
    return s_e, s_t, s_h, s_dow, (s_rt, y_q)

def get_eps_greedy_actionV(Qs, hp_Q, i, seed=32):
    np.random.seed(seed)
    # choose action
    acts = np.argmax(Qs, axis=1)
    epsi = hp_Q['eps']*np.exp(-i*hp_Q['eps_dec'])
    eps_idx = np.random.uniform(size=len(Qs)) < epsi
    
    acts[eps_idx] = np.random.choice(range(3), size=eps_idx.sum(), replace=True, 
                                     p=[(1-hp_Q['pct_c'])/2, hp_Q['pct_c'], (1-hp_Q['pct_c'])/2])
    return acts

def get_charge(e, a, b_params, return_eff=False):
    if a == 0:
        out = [0, 0]
    elif a == 1:
        out = [np.minimum(b_params['capacity']/4, b_params['dur']*b_params['capacity'] - e), 1/b_params['efficiency']]
    else:
        out = [-np.minimum(b_params['capacity']/4, e), b_params['efficiency']/4]

    if return_eff:
        return out
    return out[0]

def get_chargeV(e, a, b_params, return_eff=False):
    charge = np.zeros(len(a))
    eff = np.zeros(len(a))
    charge[a == 1] = np.minimum(b_params['capacity']/4, b_params['dur']*b_params['capacity'] - e)
    charge[a == 2] = -np.minimum(b_params['capacity']/4, e)
    
    if not return_eff:
        return charge
    
    eff[a == 1] = 1/b_params['efficiency']
    eff[a == 2] = b_params['efficiency']
    return charge, eff

def R(e, t, a, y, yma, hp, b_params): # original version of reward function
    pma = yma[t]
    p = y[t]
    charge, eff = get_charge(e*b_params['capacity']/4, a, b_params, True) 
    
    r = (1-hp['pct_rev'])*hp['pct_ma'] * ((pma - p)*(charge))
    r += (1-hp['pct_rev'])*(1-hp['pct_ma']) * (e*b_params['capacity']/4 + charge)
    r += hp['pct_rev'] * (p*(-charge*eff))
    return r

def RV(e, a, y, yma, hp, b_params): # original version of reward function
    pma = yma
    p = y
    charge, eff = get_chargeV(e*b_params['capacity']/4, a, b_params, True) 
    
    r = (1-hp['pct_rev'])*hp['pct_ma'] * ((pma - p)*(charge))
    r += (1-hp['pct_rev'])*(1-hp['pct_ma']) * (e*b_params['capacity']/4 + charge)
    r += hp['pct_rev'] * (p*(-charge*eff))
    return r

def Q_updateV(Q, y, yma, states, b_params, hp_Q, hp_R, i, chunk_size):
    s_e, s_t, s_h, s_dow, s_rt = states
    t = 0
    
    while t < len(s_rt)-1:
        
        tt = t + chunk_size
        
        for e in tqdm(range(len(s_e)), disable=True):
            a = get_eps_greedy_actionV(Q[e, s_t[:-1][t:tt], s_h[:-1][t:tt], s_dow[:-1][t:tt], s_rt[:-1][t:tt], :], hp_Q, i)
            sp_e = np.clip(e + A_LOOKUP[a], 0, len(s_e)-1)
            r = RV(e, a, y[:-1][t:tt], yma[:-1][t:tt], hp_R, b_params)

            Q[e, s_t[:-1][t:tt], s_h[:-1][t:tt], s_dow[:-1][t:tt], s_rt[:-1][t:tt], a] = (
                    (1 - hp_Q['alpha']) * Q[e, s_t[:-1][t:tt], s_h[:-1][t:tt], s_dow[:-1][t:tt], s_rt[:-1][t:tt], a] + 
                    hp_Q['alpha'] * r + 
                    hp_Q['alpha'] * hp_Q['gamma'] * np.max(Q[sp_e, s_t[1:][t:tt], s_h[1:][t:tt], 
                                                             s_dow[1:][t:tt], s_rt[1:][t:tt], :])
                )
        t = tt
    return Q


def train_Q(states, y, b_params, hp_Q, hp_R, iters=Q_ITERS, chunk_size=Q_CHUNKSIZE, verbose=False, eps=1e-16):
    # Q iterate batch
    s_e, s_t, s_h, s_dow, s_rt = states
    yma = pd.Series(y).ewm(alpha=hp_R['alpha']).mean()
    Q = np.zeros([len(s_e), np.max(s_t)+1, np.max(s_h)+1, np.max(s_dow)+1, np.max(s_rt)+1, 3])
    eps_log = np.zeros(iters)
    Q_log = np.zeros(iters)

    for i in tqdm(range(iters), disable=not verbose):
        Qi = Q_updateV(Q, y, yma, states, b_params, hp_Q, hp_R, i, chunk_size)
        Q_log[i] = np.abs(Qi/(Q+eps) - 1).mean()
        Q = Qi
    
    return Q, Q_log, eps_log


def validate_Q(Q, states, y, b_params, hp_R, verbose=False):
    s_e, s_q, s_h, s_dow, s_rt = states
    yma = pd.Series(y).ewm(alpha=hp_R['alpha']).mean()
    N = len(y)
    a = np.zeros(N, dtype=int)
    c = np.zeros(N)
    e = np.zeros(N, dtype=int)
    revenue = np.zeros(N)
    reward = np.zeros(N)

    for t in tqdm(range(N-1), disable=not verbose):
        # get policy
        a[t] = np.argmax(Q[e[t], s_q[t], s_h[t], s_dow[t], s_rt[t], :])
        # save outcomes
        c[t], eff = get_charge(e[t]*b_params['capacity']/4, a[t], b_params, True)
        e[t+1] = np.clip(e[t] + A_LOOKUP[a[t]], 0, len(s_e)-1)
        revenue[t] = s_rt[t] * (-c[t]*eff)
        reward[t] = R(e[t], t, a[t], y, yma, hp_R, b_params)

    ace = np.array([a, c, e])
    revrew = np.array([revenue, reward])
    
    return ace, revrew

def get_Q_hyperparameters(node, dur, hp_override):
    # set fixed hypterparameters
    hp_Q = {'pct_c':0.5, 'alpha':0.3, 'gamma':0.95, 
        'eps': 0.99, 'eps_dec': 0.1, 'm':100}
    hp_R = {'alpha':0.5, 'pct_rev':0.5, 'pct_ma':1}
    
    # update with best hyperparameters
    if hp_override is None:
        summ = pd.read_csv(f'{PATH_RESULTS}gsQ_summ_N{node.lower()}_D{dur}.csv')
        summmean = summ.groupby(['alpha', 'epsilon', 'pct_rev', 'pct_ma']).mean().reset_index()
        opt_idx = summmean.cumrev.argmax()
        hp_optimal = summmean.iloc[opt_idx].values[:4]
    else:
        hp_optimal = hp_override
    
    hp_Q['alpha'] = hp_optimal[0]
    hp_Q['eps'] = hp_optimal[1]
    hp_R['pct_rev'] = hp_optimal[2]
    hp_R['pct_ma'] = hp_optimal[3]
    
    return hp_Q, hp_R