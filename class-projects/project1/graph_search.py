import numpy as np
import networkx as nx
import pandas as pd
import time
import sys
from tqdm import tqdm

from utils import get_bscore
from utils import is_cyclic, move_to_rand_neighbor
from utils import read_data, save_results

def k2_search(nodes, data, max_par_nodes=None):
    time_total = -time.time()
    
    if max_par_nodes is None:
        max_par_nodes = len(nodes)
        
    # initialize graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    bscore_best = get_bscore(G, data, nodes)
    
    # first node is oldest anscestor, loop through all potential children
    for i, chi_node in enumerate(nodes[1:]):
        bscore_cand, par_node_cand = -np.inf, np.nan
        par_node_count = 0

        # loop through all potential parents of a given child
        # add new edges that improve bscore until either 
        # max nodes are hit or there aren't any more parents
        while par_node_count < np.min((max_par_nodes, len(nodes[0:i+1]))):
            for par_node in nodes[0:i+1]:
                if not G.has_edge(par_node, chi_node):
                    G.add_edge(par_node, chi_node)
                    bscore_cand_trial = get_bscore(G, data, nodes)
                    if bscore_cand_trial > bscore_cand:
                        bscore_cand = bscore_cand_trial
                        par_node_cand = par_node
                    G.remove_edge(par_node, chi_node)

            # add the best parenta
            if bscore_cand > bscore_best:
                bscore_best = bscore_cand
                par_node_count += 1
                G.add_edge(par_node_cand, chi_node)
            else: 
                break
    
    time_total += time.time()
                
    return G, bscore_best, time_total


def local_search(G, nodes, data, max_iter=25):
    time_total = -time.time()
    bscore_best = get_bscore(G, data, nodes)
    iters = 0
    
    while iters < max_iter:
        G_new = move_to_rand_neighbor(G)
        
        if is_cyclic(G_new):
            bscore_cand = -np.inf
        else:
            bscore_cand = get_bscore(G_new, data, nodes)
            
        if bscore_cand > bscore_best:
            bscore_best = bscore_cand
            G = G_new
            iters = 0
        else:
            iters += 1
        
    time_total += time.time()
            
    return G, bscore_best, time_total


def random_start_graphsearch(nodes, data, iterations=3):
    time_total = -time.time()
    time_mean_iter = 0
    bscore_best, G_best = -np.inf, np.nan
    
    for i in tqdm(range(iterations)):
        time_mean_iter -= time.time()
        
        # shuffle nodes
        rs_idx = np.random.choice(np.arange(len(nodes)), size=len(nodes), replace=False)
        rs_nodes = [nodes[r] for r in rs_idx]
        rs_data = data[:,rs_idx]
        
        # run k2 search
        G_cand, bscore_cand, __ = k2_search(rs_nodes, rs_data)
        # run local search
        G_cand, bscore_cand, __ = local_search(G_cand, rs_nodes, rs_data)
        
        if bscore_cand > bscore_best:
            bscore_best = bscore_cand
            G_best = G_cand
        
        time_mean_iter += time.time()
        
    time_total += time.time()
    time_mean_iter /= iterations
    
    return G_best, bscore_best, (time_total, time_mean_iter)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <filename prefix> n_iterations")
    
    filename = sys.argv[1]
    n_iterations = int(sys.argv[2])
    results = {}
    
    # read data
    print('reading data...')
    nodes, df = read_data(filename)

    # get graph with best bayescore
    print('getting best graph...')
    G, bayescore, total_times = random_start_graphsearch(nodes, df, iterations=n_iterations)
    
    # save results
    print('saving results...')
    save_results(G, filename)

    # print table output
    results['Bayes score'] = [np.round(bayescore, 3)]
    results['Graph structure (N, E)'] = [f'({len(G.nodes)}, {len(G.edges)})']
    results['Total time (sec)'] = [np.round(total_times[0], 3)]
    results[f'Mean iteration time (sec, {n_iterations} iters)'] = [np.round(total_times[1])]
    print(pd.DataFrame(results).T.to_latex())
    


if __name__ == '__main__':
    main()