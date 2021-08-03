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
    while (max(abs(mu_next - mu)) >= eps):
        mu = np.copy(mu_next)
        mu_next = alpha * mu @ P + (1 - alpha) * uniform
    return mu

def reverse_graph(graph):
    graph = np.array(graph)
    ret_graph = np.zeros_like(graph)
    n, m = graph.shape
    assert (n == m)
    for i in range(n):
        for j in range(m):
            if graph[i, j]:
                ret_graph[j, i] = 1
    return ret_graph

def generate_transfer_matrix(graph, corr):
    """generate transfer matrix given graph"""
    # graph = reverse_graph(graph)
    graph = np.array(graph)
    corr = np.array(corr)
    assert graph.shape == corr.shape
    n = graph.shape[0]

    P = np.zeros_like(graph)
    for i in range(n):
        w = 0.0
        for j in range(n):
            if graph[i, j]:
                w += abs(corr[i, j])
        for j in range(n):
            if graph[i, j]:
                P[i, j] = abs(corr[i, j]) / w
    return P

