import numpy as np
import matplotlib.pyplot as plt
from utils_battery import get_optimal_battery_schedule, get_efficiency
from utils_data import NODES
import statsmodels.api as sm

from utils_data import PATH_HOME

def set_plt_settings():
    plt.rcParams.update({'font.size': 14})
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


def plot_optimal_performance(p, dt_start, duration, capacity, use_efficiency=True):
    t = len(p)
    e, c, d, revenue = get_optimal_battery_schedule(p, duration=duration, charge_capacity=capacity, use_efficiency=use_efficiency)
    schedule = c - d

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
    ax1r = ax[1].twinx()
    ax[0].set_title(f'{int(t/4)}-hr / {int(t/4/24)}-day / {np.round(t/4/24/365, 1)}-yr optimization\n' + 
                    f'Duration: {duration}hrs; Capacity: {capacity}MW; Round-trip efficiency: {use_efficiency}\nRevenue: ${rev:,.2f}')
    ax[0].set_ylabel('Energy storage (MWh)\nDis/charge rate (MW)')
    ax[1].set_xlabel(f'hrs from {dt_start}')
    ax1r.set_ylabel('Price ($/MWh)')
    ax[1].set_ylabel('Revenue ($)')
    ax[0].step(np.arange(t)/4, schedule, alpha=0.75, color='red', label='charge/discharge schedule')
    ax1r.plot(np.arange(t)/4, p, alpha=0.15, color='grey', label='spot prices')
    ax[0].plot(np.arange(t)/4, e, alpha=0.75, label='storage')
    ax[1].plot(np.arange(t)/4, revenue, alpha=0.75, color='green', label='revenue')
    ax[0].legend()
    ax[1].legend()
    plt.show()

def plot_optimal_rev_by_duration(p, dt_start, durations, capacity, use_efficiency=True):
    t = len(p)
    fig, ax0 = plt.subplots(figsize=(10,5))
    ax1 = ax0.twinx()
    ax1.plot(np.arange(t)/4, p, color='grey', alpha=0.15)
    for dur in durations:
        __, c, d, revenue = get_optimal_battery_schedule(p, duration=dur, charge_capacity=capacity, 
                                                    use_efficiency=use_efficiency)

        ax0.plot(np.arange(t)/4, revenue, alpha=0.75, label=f'{dur}hrs')

    ax0.set_title(f'{int(t/4)}-hr / {int(t/4/24)}-day / {np.round(t/4/24/365, 1)}-yr optimization, by duration\n' + 
                  f'Capacity={capacity}; Round-trip efficiency={use_efficiency}')
    ax0.set_ylabel('Revenue ($)')
    ax0.set_xlabel(f'hours since {dt_start}')
    ax1.set_ylabel('Price ($/MWh)')
    ax0.legend()
    plt.show()
    
    
def plot_ts_model(tmodel, X, y, node=NODES[0], t=1000):# get plot data
    ypred_val = tmodel.predict(X).values
    acorr = sm.tsa.acf(y - ypred_val)
    n_mask = X.loc[:, f'node_{node.upper()}'] == 1

    # look at autocorrelation
    fig, ax = plt.subplots(nrows=3, figsize=(4, 7), constrained_layout=True)
    ax[0].plot(acorr)
    ax[1].scatter(y, y - ypred_val, alpha=0.5)

    ax[2].plot(np.arange(t)/4/24, y[n_mask][:t], alpha=0.5, label='true')
    ax[2].plot(np.arange(t)/4/24, ypred_val[n_mask][:t], alpha=0.5, label='pred')

    ax[0].set_title('Autocorrelation of errors')
    ax[1].set_title('Errors against price')
    ax[2].set_title(f'Actual and pred. prices')
    ax[0].set_xlabel('lag')
    ax[0].set_ylabel('autocorrelation')
    ax[1].set_ylabel('prediction errors')
    ax[1].set_xlabel('price ($/MW)')
    ax[2].set_xlabel('days in 2022')
    ax[2].set_ylabel('price ($/MW)')
    ax[2].legend()

    plt.savefig(PATH_HOME + '/results/' + 'timeseries.png', pad_inches=0.5)
    plt.show()


def plot_Qval(ace, revrew, y, b_params, yr):
    __, ax = plt.subplots(nrows=2, figsize=(10, 7))
    ax1r = ax[1].twinx()
    ax[0].plot(ace[2]*b_params['capacity']/4, label='energy')
    ax1r.plot(y, color='grey', alpha=0.2)
    ax[1].plot(np.cumsum(revrew[0]), label='cumm. revenue', color='green')
    ax[1].plot(np.cumsum(revrew[1]), label='cumm. reward')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title(f'Battery operation in {yr}')
    ax[1].set_title(f'Total revenue={int(revrew[0].sum())}, Total reward={int(revrew[1].sum())}')
    plt.show()