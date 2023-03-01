import numpy as np
import cvxpy as cp
import datetime as dt

def get_efficiency(duration):
    if duration < 12:
        return 0.925
    if duration < 48:
        return 0.86
    return 0.70


def get_optimal_battery_schedule(px:np.array, duration:int, charge_capacity:int, storage_start=0., use_efficiency=True):
    t = len(px)
    energy_capacity = duration*charge_capacity
    if use_efficiency:
        efficiency = get_efficiency(duration)
    else:
        efficiency = 1

    
    
    # variables
    c = cp.Variable(t) # charging at time t
    d = cp.Variable(t) # dischargint at time t
    e = cp.Variable(t) # energy at time t
    # constraints
    constraints = [e[1:] == e[:t-1] + c[:t-1]/4 - d[:t-1]/4] # evolution of energy over time, divide by 4 since 15min intervals
    constraints += [e[0] == storage_start] # must start at 0
    # constraints += [e[t-1] == 0] # must end at 0
    constraints += [e <= energy_capacity, e >= 0] # energy capacity requirements
    constraints += [c <= charge_capacity, c >= 0] # power capacity requirements
    constraints += [d <= charge_capacity, d >= 0] # power capacity requirements
    # problem
    obj = cp.Maximize(px @ (efficiency*d/4 - (1/efficiency)*c/4)) # divide by 4 since 15min intervals
    prob = cp.Problem(obj, constraints)
    prob.solve()

    revenue = np.cumsum(px*(efficiency*d.value/4 - (1/efficiency)*c.value/4))
    
    return e.value, c.value, d.value, revenue

def get_limited_optimal_battery_schedule(days_foresight:float, px:np.array, duration:int, 
                                         charge_capacity:int, storage_start=0., use_efficiency=True):
    window_len = 4*24*days_foresight
    periods = int(np.ceil(len(px)/window_len))

    rev_last = 0
    e_last = 0
    revenue_optlim = np.array(())
    e_optlim, c_optlim, d_optlim = np.array(()), np.array(()), np.array(())

    for period in range(periods):
        psub = px[window_len*period:window_len*(period+1)]

        e, c, d, rev = get_optimal_battery_schedule(psub, duration, charge_capacity, 
                                                    storage_start=e_last, use_efficiency=True)

        rev += rev_last
        rev_last = rev[-1]
        e_last = e[-1]

        revenue_optlim = np.concatenate([revenue_optlim, rev])
        e_optlim = np.concatenate([e_optlim, e])
        c_optlim, d_optlim = np.concatenate([c_optlim, c]), np.concatenate([d_optlim, d])
    
    return e_optlim, c_optlim, d_optlim, revenue_optlim


def get_naive_battery_schedule(p, time, duration, charge_capacity, use_efficiency=True):
    if use_efficiency:
        efficiency = get_efficiency(duration)
    else:
        efficiency = 1
    
    # make charge/discharge schedule
    charge_mask = (
        ((time >= dt.time(2, 0)) & (time < dt.time(4, 0))) | 
        ((time >= dt.time(11,0)) & (time < dt.time(15,0))))
    discharge_mask = (
        (time >= dt.time(8, 0)) & (time < dt.time(10, 0)) | 
        ((time >= dt.time(17,0)) & (time < dt.time(21,0))))
    
    c_sched = np.zeros(len(time))
    c_sched[charge_mask] = 1*charge_capacity/4
    c_sched[discharge_mask] = -1*charge_capacity/4
    
    # define energy
    e = np.zeros(len(p))
    for i in range(1, len(p)):
        e[i] = np.minimum(duration*charge_capacity, np.maximum(0, e[i-1] + c_sched[i-1]))
        
    # calculate revenue
    d = -1*np.minimum(0, e[1:] - e[:-1])
    c = np.maximum(0, e[1:] - e[:-1])
    revenue = p[:-1] * (d*efficiency - c/efficiency)
    revenue_cum = np.cumsum(revenue)
    
    return e, c, d, revenue_cum