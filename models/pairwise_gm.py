
import numpy as np

from models.utils import prod_ev
from models.utils import sum_ev, log_sum_exp

class graphical_model(object):
    """
    loopy pairwaise markov random field
    """

    def __init__(self, nodes, type=None):
        """initialize the model with the input nodes, interractions between nodes may already exist"""

        self.nodes = nodes
        self.size = len(nodes)
        self.type = type

    def init_messages(self, init="ones", decode_type="SPA"):
        if init == "ones":
            if decode_type == "SPA" or decode_type == "MPA":
                self.messages = [np.ones((len(ne.neighbours),  ne.K)) for ne in self.nodes]
            elif decode_type == "MSA":
                self.messages = [np.zeros((len(ne.neighbours),  ne.K)) for ne in self.nodes]

    def init_loopy(self, decode_type="SPA"):
        if decode_type == "SPA" or decode_type == "MPA":
            self.inter = [ne.compute_inter_matrix() for ne in self.nodes]
        elif decode_type == "MSA":
            self.log_inter = [ne.compute_log_inter_matrix() for ne in self.nodes]

    def register(self):
        self.register_ = np.empty((self.size, self.size))
        self.register_[:] = np.nan
        for i, node in enumerate(self.nodes):
            for j, ne in enumerate(node.neighbours):
                self.register_ [i, ne.label] = j

    def loop(self, sample, decode_type="SPA"):
        """parallel loopy belief propagation"""
        new_messages = [np.zeros((len(ne.neighbours), ne.K)) for ne in self.nodes]

        for ii, node in enumerate(self.nodes):
            if decode_type == "SPA" or decode_type == "MPA":
                prod_ = prod_ev(self.messages[ii])
            elif decode_type == "MSA":
                sum_ = sum_ev(self.messages[ii])
                #if ii == 0:
                #    print("sum")
                #    print(sum_)
            if not(np.isnan(sample[ii])):
                j = np.argmax(node.values == sample[ii])
                if decode_type == "SPA" or decode_type == "MPA":
                    message = np.diag(prod_[:, j]) @ self.inter[ii][:, :, j]
                for i, ne in enumerate(node.neighbours):
                    if decode_type == "MSA":
                        message = self.log_inter[ii][i, :, j] + sum_[i, j]
                        new_messages[ne.label][int(self.register_[ne.label, node.label])] = message - log_sum_exp(message)
                    elif decode_type == "SPA" or decode_type == "MPA":
                        #if ne.label == 0:
                        #    print(self.register_[ne.label, node.label])
                        new_messages[ne.label][int(self.register_[ne.label, node.label])] = message[i] / np.sum(message[i])
            else:
                for i, ne in enumerate(node.neighbours):
                    if decode_type == "SPA" or decode_type == "MPA":
                        message = self.inter[ii][i] @ np.diag(prod_[i])
                    elif decode_type == "MSA":
                        message = self.log_inter[ii][i] + np.repeat(sum_[i][None, :], ne.K, axis=0)
                    if decode_type == "SPA":
                        message = np.sum(message, axis=1)
                    elif decode_type == "MPA" or decode_type == "MSA":
                        message = np.max(message, axis=1)
                    if decode_type == "SPA" or decode_type == "MPA":
                        new_messages[ne.label][int(self.register_[ne.label, node.label])] = message / np.sum(message)
                    elif decode_type == "MSA":
                        #if ne.label == 0:
                        #    print(self.register_[ne.label, node.label])
                        new_messages[ne.label][int(self.register_[ne.label, node.label])] = message - log_sum_exp(message)

        change = np.array([not(np.allclose(new_messages[i], self.messages[i])) for i in range(len(self.nodes))]).any()
        self.messages = new_messages
        #print("message")
        #print(self.messages[0])
        #print(self.messages[0])
        return change

    def update_distrib(self, sample, decode_type="SPA"):
        for i, node in enumerate(self.nodes):
            node.distribution(self.messages[i], sample[i], decode_type)
