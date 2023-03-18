import pandas as pd
import numpy as np
import sys, os

from utils_data import make_analysis_dataset, NODES, PATH_DATA
from utils_battery import get_efficiency
from utils_cv import train_test_split, crossval_model

# define file locations
PATH_RESULTS = f'{PATH_DATA}results/'
if not os.path.isdir(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)

def main():
    if len(sys.argv) != 3:
            raise Exception("usage: python learn_modelbased_approach.py <node index> <duration>")
    node_idx = int(sys.argv[1])
    dur = int(sys.argv[2])

    # read in data
    df = make_analysis_dataset(nodes=[NODES[node_idx]])

    # set columns
    mark_cols = ['lmp_rt_m1', 'lmp_rt_m2'] + ['lmp_da', 'hour', 'quarter', 'weekday']
    node_cols = [col for col in df.columns if col.startswith(('node'))] 
    X_cols = mark_cols + node_cols + ['lmp_rt_m1_rolling']
    y_col = 'lmp_rt'
    group_cols = ['year']

    # set params
    b_params = {'dur':dur, 'capacity':200}
    b_params['efficiency'] = get_efficiency(b_params['dur'])
    S = 3000
    kmax=100

    # grid search
    hp_weight_notrev = [0.25, 0.5, 0.66, 1.]
    hp_weight_split = [0.5, 1.]
    hp_ma = [0.25, 0.5, 0.8]
    gssum = {}
    gssummean = {}
    for wr in hp_weight_notrev:
        for ws in hp_weight_split:
            hp_weights = [wr*ws, wr*(1-ws)]
            for ma in hp_ma:
                # redefine rolling period
                df['lmp_rt_m1_rolling']  = df.lmp_rt_m1.ewm(com=ma).mean()
                # split data
                X_tt, y_tt, g_tt, __, __ = train_test_split(df, X_cols, y_col, group_cols, yr_val=2022)
                # cross validate
                label = f'weights={hp_weights + [1-wr]}, ma={ma}'
                cvsum, cvsummean = crossval_model(X_tt, y_tt, g_tt, hp_weights, b_params, S, kmax, desc=label)
                print('mean revenue:', cvsummean['cumrev'])
                gssum[(wr*ws, wr*(1-ws), ma)] = cvsum
                gssummean[(wr*ws, wr*(1-ws), ma)] = cvsummean

    # save logs
    summ = pd.DataFrame([])
    for k, v in gssum.items():
        d = pd.DataFrame(v)
        d['w_roll'], d['w_e'], d['ma'] = k[0], k[1], k[2]
        summ = pd.concat([d, summ])
    summ.to_csv(f'{PATH_RESULTS}hptune_summ_{NODES[node_idx].lower()}_{dur}.csv', index=False)

if __name__ == '__main__':
    main()