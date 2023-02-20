import numpy as np
import matplotlib.pyplot as plt
from battery_utils import get_optimal_battery_schedule, get_rt_efficiency

def set_plt_settings():
    plt.rcParams.update({'font.size': 14})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_optimal_performance(p, dt_start, duration, capacity, use_efficiency=True):
    t = len(p)
    e, c, d, rev = get_optimal_battery_schedule(p, duration=duration, charge_capacity=capacity, use_efficiency=use_efficiency)
    schedule = c - d
    if use_efficiency:
        revenue = np.cumsum(p*(get_rt_efficiency(duration)*d/4 - c/4))
    else:
        revenue = np.cumsum(p*(d/4 - c/4))

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
        __, c, d, __ = get_optimal_battery_schedule(p, duration=dur, charge_capacity=capacity, 
                                                    use_efficiency=use_efficiency)
        if use_efficiency:
            revenue = np.cumsum(p*(get_rt_efficiency(dur)*d/4 - c/4))
        else:
            revenue = np.cumsum(p*(d/4 - c/4))
        ax0.plot(np.arange(t)/4, revenue, alpha=0.75, label=f'{dur}hrs')

    ax0.set_title(f'{int(t/4)}-hr / {int(t/4/24)}-day / {np.round(t/4/24/365, 1)}-yr optimization, by duration\n' + 
                  f'Capacity={capacity}; Round-trip efficiency={use_efficiency}')
    ax0.set_ylabel('Revenue ($)')
    ax0.set_xlabel(f'hours since {dt_start}')
    ax1.set_ylabel('Price ($/MWh)')
    ax0.legend()
    plt.show()