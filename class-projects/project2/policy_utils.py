import numpy as np
from tqdm import tqdm
import scipy as sp


def lookahead(U, gamma, s, a, r, T):
    for i in range(len(s)):
        U[s[i]] = r[i] + gamma * T[a[i]][:, s[i]].T @ U
    return U

def backup(U, gamma, s, n_state, n_act, R, T):
    Q_sa = np.zeros((n_state, n_act))
    for ai in range(n_act):
        Q_sa[:, ai] = lookahead(U, gamma, s, 
                                np.repeat(ai, len(s)), 
                                R[s, ai],
                                T)
    return Q_sa

def policy_evaluation(U, gamma, s, policy, R, T, k_max=100, tol=0.1):
    Up = U.copy()
    for k in range(k_max):
        Up = lookahead(Up, gamma, s, policy[s], R[s, policy[s]], T)
        max_diff = np.max(np.abs(Up - U))
        if max_diff < tol:
            break
        else:
            U = Up.copy()
    return U

def policy_improvement(U, gamma, s, n_state, n_act, R, T):
    return np.argmax(backup(U, gamma, s, n_state, n_act, R, T), axis=1)

def policy_iteration(gamma, s, R, T, n_state, n_act, k_max=5, tol=3):
    U = np.zeros(n_state)
    policy = np.random.randint(n_act, size=n_state)
    for k in tqdm(range(k_max)):
        U = policy_evaluation(U, gamma, s, policy, R, T, k_max=100)
        policyp = policy_improvement(U, gamma, s, n_state, n_act, R, T)
        max_diff = np.max(np.abs(policyp - policy))
        if max_diff < tol:
            break
        else:
            policy = policyp.copy()
    return policy, U

def Q_learn(gamma, s, a, r, spr, n_state, n_act, k_max=100, one_shot=False):
    Q = np.zeros((n_state, n_act))
    policy_log = np.zeros((k_max, n_state))
    for i in tqdm(range(k_max)):
        if one_shot:
            Q[s, a] += (1/k_max) * gamma * (r + np.max(Q[spr], axis=1) - Q[s, a])
        else:
            for j in range(len(s)):
                Q[s[j], a[j]] += (1/k_max) * gamma * (r[j] + np.max(Q[spr[j]]) - Q[s[j], a[j]])
        policy_log[i] = np.argmax(Q, axis=1)
    
    return Q, policy_log

def get_missing_states(n_state, s):
    possible_states = np.arange(n_state)
    sampled_states = np.where(np.in1d(possible_states, s))[0]
    missing_states = [i for i in range(n_state) if i not in sampled_states]
    return sampled_states, missing_states

def get_medium_substates(s):
    spos = np.mod(s, 500)
    svel = ((s - spos)/500).astype(int)
    return spos, svel

def get_medium_state(spos, svel):
    return (500*svel + spos).astype(int)

def get_mle_problem(s, a, r, spr, n_state, n_act):
    # estimate transition probabilities and reward
    # get state/action counts
    N_sa = np.zeros((n_state, n_act), dtype=float)
    for i in range(len(s)):
        N_sa[s[i], a[i]] += 1.0
    
    print('building dictionary of sparse matrices...')
    T_asps = {}
    for ai in range(n_act):
        T_asps[ai] = sp.sparse.csr_matrix((n_state, n_state), dtype=float)
    R_sa = np.zeros((n_state, n_act), dtype=float)
    
    print('calculating MLE...')
    for i in tqdm(range(len(s))):
        T_asps[a[i]][spr[i], s[i]] += 1/N_sa[s[i], a[i]]
        R_sa[s[i], a[i]] += r[i]/N_sa[s[i], a[i]]
    
    return T_asps, R_sa, N_sa