
import numpy as np

from models.utils import prod_ev
from scipy.special import softmax

class node_(object):

        def __init__(self, label, values, potential=lambda x: 1., neighbours=[],
                     potential_neighbours=[]):
            """
            params:
            values : list of values that can take the node
            potential : potential \\psi_j of the node
            neighbours: list of neighbours in the graphical model
            potential_neighbours: list of \\psi_{i,j} for all neighbour i
            """

            self.label = label
            self.values = values
            self.K = len(values)
            self.potential = potential

            assert len(neighbours) == len(potential_neighbours)
            self.neighbours = neighbours
            self.potential_neighbours = potential_neighbours


        def distribution(self, messages, x_true=np.nan, decode_type="SPA", normalize=False):
            """update the marginal distribution of the node"""
            if not(np.isnan(x_true)):
                self.distr = np.zeros(len(self.values))
                self.distr[np.array(self.values) == x_true] = 1.
            elif self.neighbours != []:
                res = np.zeros(self.K)
                for i, k in enumerate(self.values):
                    if decode_type == "SPA" or decode_type == "MPA":
                        res[i] = np.prod(messages[:, i]) *  self.potential(k)
                    elif decode_type == "MSA":
                        res[i] = np.sum(messages[:, i]) + self.potential(k)
                if decode_type == "SPA":
                    self.distr = res
                    if normalize:
                        self.distr /= np.sum(self.distr)
                elif decode_type == "MPA" or decode_type == "MSA":
                    #print(res)
                    self.distr = np.zeros(res.shape)
                    self.distr[np.argmax(res)] = 1.
            else:
                self.distr = np.empty(self.K)


        def add_neighbour(self, node, potential):
            """add a neighbour to the node, with a potential describing the interaction between both nodes"""
            if node not in self.neighbours:
                self.neighbours.append(node)
                self.potential_neighbours.append(potential)


        def compute_inter_matrix(self):
            """only works with all nodes share the same number of possible values K
            potentials (nodes, values of nodes, values of self)
            """
            inter_matrix = np.empty((len(self.neighbours), self.K, self.K))
            for i, node in enumerate(self.neighbours):
                for ji, xi in enumerate(node.values):
                    for jj, xj in enumerate(self.values):
                        inter_matrix[i, ji, jj] = self.potential(xj) \
                        * self.potential_neighbours[i](xj, xi)
            return inter_matrix

        def compute_log_inter_matrix(self):
            log_inter_matrix = np.empty((len(self.neighbours), self.K, self.K))
            for i, node in enumerate(self.neighbours):
                for ji, xi in enumerate(node.values):
                    for jj, xj in enumerate(self.values):
                        log_inter_matrix[i, ji, jj] = self.potential(xj) \
                        + self.potential_neighbours[i](xj, xi)
            return log_inter_matrix
