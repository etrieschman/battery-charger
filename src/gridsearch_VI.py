import pandas as pd
import numpy as np
import sys, os
from datetime import datetime as dt
from utils_data import make_analysis_dataset, NODES, PATH_DATA
from utils_battery import get_efficiency
from utils_mdp_m import R, R2
from utils_cv import train_test_split, crossval_VI

# define file locations
PATH_RESULTS = f'{PATH_DATA}results/'
if not os.path.isdir(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
PATH_RESULTS_LOG = f'{PATH_RESULTS}log/'
if not os.path.isdir(PATH_RESULTS_LOG):
    os.makedirs(PATH_RESULTS_LOG)


def learn_modelbased_vi(node_idx, dur, hp_weight_rev, hp_store, hp_ma, R, S, logsuff):
    # if len(sys.argv) != 3:
    #         raise Exception("usage: python learn_modelbased_approach.py <node index> <duration>")
    # node_idx = int(sys.argv[1])
    # dur = int(sys.argv[2])

    # read in data
    df = make_analysis_dataset(nodes=[NODES[node_idx]])

    # set columns
    mark_cols = [col for col in df.columns if col.startswith(('lmp_rt_m'))] + ['lmp_da']
    node_cols = [col for col in df.columns if col.startswith(('node'))] 
    time_cols = [col for col in df.columns if col.startswith(('h_'))] + ['weekday']
    XOLS_cols = mark_cols + time_cols + node_cols + ['lmp_rt_m1_rolling']
    Xvi_cols = ['lmp_rt_m1', 'lmp_rt_m2', 'lmp_rt_m1_rolling', 'lmp_da'] + time_cols
    y_col = 'lmp_rt'
    group_cols = ['year']

    # set params
    b_params = {'dur':dur, 'capacity':200}
    b_params['efficiency'] = get_efficiency(b_params['dur'])
    kmax=50

    # grid search
    gssum = {}
    gssummean = {}
    for wr in hp_weight_rev:
        for ws in hp_store:
            hp = [wr, ws]
            for ma in hp_ma:
                # redefine rolling period
                df['lmp_rt_m1_rolling']  = df.lmp_rt_m1.ewm(alpha=ma).mean()
                # split data
                X_tt, y_tt, g_tt, __, __ = train_test_split(df, XOLS_cols, y_col, group_cols, yr_val=2022)
                # cross validate
                label = f'wrev={wr}, wstor={ws}, ma={ma}'
                cvsum, cvsummean = crossval_VI(X_tt, Xvi_cols, y_tt, g_tt, hp, b_params, R, S, kmax, desc=label)
                print('mean revenue:', cvsummean['cumrev'])
                gssum[(wr, ws, ma)] = cvsum
                gssummean[(wr, ws, ma)] = cvsummean

    # save logs
    summ = pd.DataFrame([])
    for k, v in gssum.items():
        d = pd.DataFrame(v)
        d['w_roll'], d['w_e'], d['ma'] = k[0], k[1], k[2]
        summ = pd.concat([d, summ])
    summ.to_csv(f'{PATH_RESULTS}hptune_summ_{NODES[node_idx].lower()}_{dur}.csv', index=False)
    summ.to_csv(f'{PATH_RESULTS_LOG}hptune_summ_{NODES[node_idx].lower()}_{dur}_{logsuff}.csv', index=False)


if __name__ == '__main__':

    node_idx = 0
    
    # 4-hr duration
    # hp_weight_rev = [0.25, 0.5, 0.75, 1.]
    # hp_store = [0., 5., 25.]
    # hp_ma = [0.5, 0.9]
    # learn_modelbased_vi(node_idx, 4, 
    #                     hp_weight_rev, hp_store, hp_ma, R, None,
    #                     logsuff=dt.now().date().strftime("%Y%m%d"))

    # 100-hr duration
    hp_weight_rev = [0., 0.25, 0.5, 0.66]
    hp_store = [5., 1.]
    hp_ma = [0.5, 0.9]
    learn_modelbased_vi(node_idx, 100, 
                        hp_weight_rev, hp_store, hp_ma, R2, 5000,
                        logsuff=dt.now().date().strftime("%Y%m%d"))
    

