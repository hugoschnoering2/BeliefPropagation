
import numpy as np

from models.node import node_
from models.pairwise_gm import graphical_model

def freq_potential(freq):
    return lambda x: freq[int(x)]

def freq_edge_potential(freq):
    return lambda x, y: freq[int(x), int(y)]

def compute_freq(labels1):
    freq = np.bincount(labels1[~np.isnan(labels1)].astype(np.int))
    return freq / np.sum(freq)

def compute_bi_freq(labels1, labels2):
    nl1, nl2 = int(np.amax(labels1[~np.isnan(labels1)])+1), int(np.amax(labels2[~np.isnan(labels2)])+1)
    bi_freq = np.zeros((nl1, nl2))
    freq_lb1 = compute_freq(labels1)
    for i in range(nl1):
        if freq_lb1[i] == 0:
            pass
        else:
            index_to_keep = np.logical_and(~np.isnan(labels2), labels1==float(i))
            bi_freq[i] = np.bincount(labels2[index_to_keep].astype(np.int)) / freq_lb1[i]
    return bi_freq / np.sum(bi_freq)

def freq_pairwise_gm(G, labels, log=False):
    n_nodes = labels.shape[0]
    nodes = []
    for i in range(n_nodes):
        freq = compute_freq(labels[i])
        nodes.append(node_(i,
                           values=[float(j) for j in range(len(freq))],
                           potential=freq_potential(freq if not log else np.log(freq)),
                           neighbours=[],
                           potential_neighbours=[],
                           ))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if j != i and G[i, j] != 0:
                bi_freq = compute_bi_freq(labels[i], labels[j])
                nodes[i].add_neighbour(nodes[j], freq_edge_potential(bi_freq if not log else np.log(bi_freq)))
    gm = graphical_model(nodes)
    return gm
