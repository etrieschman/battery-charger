import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from datetime import datetime as dt
import os

from utils_data import make_analysis_dataset, NODES, PATH_DATA
from utils_cv import train_test_split, crossval_Q
from utils_battery import get_efficiency
from utils_mdp_mf import get_discrete_states

# define file locations
PATH_RESULTS = f'{PATH_DATA}results/'
if not os.path.isdir(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
PATH_RESULTS_LOG = f'{PATH_RESULTS}log/'
if not os.path.isdir(PATH_RESULTS_LOG):
    os.makedirs(PATH_RESULTS_LOG)


def gridsearch_Q(node_idx, dur, hp_grid, hp_Q, hp_R, logsuff, yr_val=2022, capacity=200):
    # make analysis datasets
    df = make_analysis_dataset(nodes=[NODES[node_idx]])
    # col params
    X_cols = ['month', 'hour', 'dow']
    y_col = 'lmp_rt'
    group_cols = ['year']

    X_tt, y_tt, g_tt, X_val, y_val = train_test_split(df, X_cols, y_col, group_cols, yr_val=yr_val)

    # set params
    b_params = {'dur':dur, 'capacity':200}
    b_params['efficiency'] = get_efficiency(b_params['dur'])
    
    # discretize data
    s_e, s_t, s_h, s_dow, (s_rt, rt_q) = get_discrete_states(X_tt, y_tt, b_params, None, hp_Q['m'])
    s_ev, s_tv, s_hv, s_dowv, (s_rtv, __) = get_discrete_states(X_val, y_val, b_params, rt_q, hp_Q['m'])
    states = s_e, s_t, s_h, s_dow, s_rt

    # grid search
    alphaQ, epsQ, pct_revR, pct_maR = hp_grid
    grid = list(itertools.product(*[alphaQ, epsQ, pct_revR, pct_maR]))
    gridcvsum = {}
    for a, e, pr, pma in tqdm(grid):
        hp_Q['alpha'] = a
        hp_Q['epsilon'] = e
        hp_R['pct_rev'] = pr
        hp_R['pct_ma'] = pma

        cvsum = crossval_Q(states, y_tt, g_tt, b_params, hp_Q, hp_R)
        gridcvsum[(a, e, pr, pma)] = cvsum
    
    # save results
    summary = (pd.concat({k: pd.DataFrame(v) for k, v in gridcvsum.items()})
           .reset_index(names=['alpha', 'epsilon', 'pct_rev', 'pct_ma', 'drop'])
           .drop(columns='drop'))
    summary.to_csv(f'{PATH_RESULTS}gsQ_summ_N{NODES[node_idx].lower()}_D{dur}.csv', index=False)
    summary.to_csv(f'{PATH_RESULTS_LOG}gsQ_summ_N{NODES[node_idx].lower()}_D{dur}_{logsuff}.csv', 
                   index=False)
    return summary


if __name__ == '__main__':
    node_idx = 0
    hp_Q = {'pct_c':0.5, 'alpha':0.3, 'gamma':0.95, 
        'eps': 0.99, 'eps_dec': 0.01, 'm':100}
    hp_R = {'alpha':0.9, 'pct_rev':0.5, 'pct_ma':0.5}

    # grid
    alphaQ = [0.1]
    epsQ = [0.5, 0.99]
    pct_revR = [0.25, 0.5, 0.75]
    pct_maR = [0., 0.5, 1.]
    hp_grid = alphaQ, epsQ, pct_revR, pct_maR

    logsuff = logsuff = dt.now().date().strftime("%Y%m%d") 
    durations = [4, 100, 6, 12, 14, 48] 
    i = 1
    for node_idx, dur in itertools.product(*[np.arange(4), durations]):
        print('---------------------------------------------------------------')
        print(f'{dt.now().strftime("%H:%M:%S")} | {i}/{len(durations)*4}: node={NODES[node_idx]}, duration={dur}')
        __ = gridsearch_Q(node_idx, dur, hp_grid, hp_Q, hp_R, logsuff)
        i += 1