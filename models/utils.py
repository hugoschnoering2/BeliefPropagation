
import numpy as np

def prod_ev(message):
    nv, K = message.shape
    prod_1 = np.ones((nv, K))
    prod_2 = np.ones((nv, K))
    for i in range(1, nv):
        prod_1[i] = prod_1[i-1] + message[i-1]
        prod_2[-(i+1)] = prod_2[-i] + message[-i]
    return prod_1 * prod_2

def sum_ev(message):
    nv, K = message.shape
    prod_1 = np.zeros((nv, K))
    prod_2 = np.zeros((nv, K))
    for i in range(1, nv):
        prod_1[i] = prod_1[i-1] + message[i-1]
        prod_2[-(i+1)] = prod_2[-i] + message[-i]
    return prod_1 + prod_2

def log_sum_exp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x-c)))
