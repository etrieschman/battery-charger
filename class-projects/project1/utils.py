import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
from tqdm import tqdm

            
def read_data(filename):
    # read in data    
    raw_data = pd.read_csv(f'./data/{filename}.csv')
    df, nodes = raw_data.values, list(raw_data.columns)
    
    return nodes, df

# draw graph
def save_results(G, filename):
    # save graph image
    plt.plot(figsize=(15, 15))
    nx.draw(G, with_labels=True)
    plt.savefig(f'./writeup/results/{filename}_path.png')
    
    # save graph file
    with open(f'{filename}.gph', 'w') as f:
        for edge in G.edges():
            f.write(f'{edge[0]}, {edge[1]}\n')
    return


def get_M(G, data, nodes, query):
    parents = list(G.predecessors(query))
    
    # get all parent instances
    max_par_insts = np.zeros(len(parents)) 
    for i, pn in enumerate(parents):
        max_par_insts[i] = np.max(data[:,nodes.index(pn)])
    par_inst = np.array(list(
        itertools.product(*[np.arange(start=1, stop=mp+1) for mp in max_par_insts])), dtype=int)

    # make M matrix and alpha matrix
    max_query_inst = np.max(data[:, nodes.index(query)])
    M = np.zeros((len(par_inst), max_query_inst), dtype=int)
    alpha = np.ones_like(M)

    # for each parent instant find data rows that exist and populate M
    par_node_idx = [nodes.index(n) for n in parents]
    query_node_idx = nodes.index(query)
    for i, pi in enumerate(par_inst):
        # get rows in data with the parent instance
        d_idx = (data[:, par_node_idx] == pi).all(axis=1)
        # get query value for those rows
        query_vals = data[d_idx, query_node_idx]
        # increment query values
        for j in query_vals:
            M[i,j-1] += 1  
    
    return M, alpha

#p 98 algorithm
def get_bscore(G, data, nodes):
    score = 0
    for query in nodes:
        M, alpha = get_M(G, data, nodes, query)
        score += np.sum(sp.special.loggamma(M+alpha))
        score -= np.sum(sp.special.loggamma(alpha))
        score += np.sum(sp.special.loggamma(np.sum(alpha, axis=1)))
        score -= np.sum(sp.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    
    return score


def move_to_rand_neighbor(G):
    n = len(G.nodes)
    i, j = np.random.choice(G.nodes(), size=2, replace=False)
    G_new = G.copy()
    
    # remove edge if there is one
    if G_new.has_edge(i, j):
        G_new.remove_edge(i, j)
        # with probability 0.5, add edge in opposite direction
        if np.random.random() > 0.5:
            G_new.add_edge(j, i)       
    # add edge if there is not one
    else:
        G_new.add_edge(i, j)
    return G_new


def is_cyclic(G):
    n_cycles = len(sorted(nx.simple_cycles(G)))
    return  n_cycles != 0

