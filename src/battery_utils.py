import numpy as np
import cvxpy as cp

def get_rt_efficiency(duration):
    if duration < 12:
        return 0.925
    if duration < 48:
        return 0.86
    return 0.70


def get_optimal_battery_schedule(px:np.array, duration:int, charge_capacity:int, use_efficiency=True):
    t = len(px)
    energy_capacity = duration*charge_capacity
    if use_efficiency:
        rt_efficiency = get_rt_efficiency(duration)
    else:
        rt_efficiency = 1
    
    # variables
    c = cp.Variable(t) # charging at time t
    d = cp.Variable(t) # dischargint at time t
    e = cp.Variable(t) # energy at time t
    # constraints
    constraints = [e[1:] == e[:t-1] + c[:t-1]/4 - d[:t-1]/4] # evolution of energy over time, divide by 4 since 15min intervals
    constraints += [e[0] == 0] # must start at 0
    # constraints += [e[t-1] == 0] # must end at 0
    constraints += [e <= energy_capacity, e >= 0] # energy capacity requirements
    constraints += [c <= charge_capacity, c >= 0] # power capacity requirements
    constraints += [d <= charge_capacity, d >= 0] # power capacity requirements
    # problem
    obj = cp.Maximize(px @ (rt_efficiency*d/4 - c/4)) # divide by 4 since 15min intervals
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return e.value, c.value, d.value, prob.value