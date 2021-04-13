
import numpy as np

from models.node import node_
from models.pairwise_gm import graphical_model

def node_potential(alpha):
    return lambda x: np.exp(alpha*x)

def log_node_potential(alpha):
    return lambda x: alpha*x

def edge_potential(beta):
    return lambda x, y: np.exp(beta*x*y)

def log_edge_potential(beta):
    return lambda x, y: beta * x * y

def ising_from_interaction_matrix(J, log=False):
    node_potential_ = node_potential if not log else log_node_potential
    edge_potential_ = edge_potential if not log else log_edge_potential
    num_nodes = J.shape[0]
    nodes_ = [node_(label=i,
                   values=[-1., 1.],
                   potential=node_potential_(J[i, i]),
                   neighbours=[],
                   potential_neighbours=[],
                   ) for i in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if j!=i and J[i, j] != 0:
                nodes_[i].add_neighbour(nodes_[j], edge_potential_(J[i, j]))
    return graphical_model(nodes_)
