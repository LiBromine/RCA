import pandas as pd
import numpy as np

def page_rank(P, alpha, eps, init=-1):
    """
    P: transition matrix,
    alpha: non-teleport proba
    init: init method, -1 means uniform, 0-N means from a specific point
    return: (maybe stationary) proba vector
    """
    n = len(P)
    uniform = np.zeros(n) + 1 / n 
    if init == -1:
        mu = np.copy(uniform)
    else:
        mu = np.zeros(n)
        mu[n] = 1

    mu_next = alpha * mu @ P + (1 - alpha) * uniform
    while (abs(mu_next - mu) >= eps):
        mu = np.copy(mu_next)
        mu_next = alpha * mu @ P + (1 - alpha) * uniform
    return mu


def Generate_Transfer_Matrix(G):
    """generate transfer matrix given graph"""
    index2node = dict()
    node2index = dict()
    for index,node in enumerate(G.keys()):
        node2index[node] = index
        index2node[index] = node
    # num of nodes
    n = len(node2index)
    # generate Transfer probability matrix M, shape of (n,n)
    M = np.zeros([n,n])
    for node1 in G.keys():
        for node2 in G[node1]:
            # FIXME: some nodes not in the Graphs.keys, may incur some errors
            try:
                M[node2index[node2],node2index[node1]] = 1/len(G[node1])
            except:
                continue
    return M, node2index, index2node

