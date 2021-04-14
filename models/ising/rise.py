import numpy as np
from numba import jit, prange
from tqdm import tqdm

@jit(nopython=True, parallel=True)
def llk(node, sample, current_params):
    res = 0
    for j in prange(len(sample)):
        if j == node:
            res -= current_params[node] * sample[node]
        else:
            res -= current_params[j] * sample[node] * sample[j]
    return np.exp(res)

@jit(nopython=True)
def grad(node, batch, current_params):
    grad = np.zeros((batch.shape[1],))
    for sample in batch:
        llk_ = llk(node, sample, current_params)
        grad += - llk_ * sample[node] * np.array([sample[j] if j!= node else 1 for j in range(batch.shape[1])])
    grad /= len(batch)
    return grad

@jit(nopython=True, parallel=True)
def prox(x, penalty, delta):
    res = np.zeros(x.shape)
    for n in prange(len(x)):
        res[n] = np.sign(x[n]) * max(np.abs(x[n])-penalty*delta, 0)
    return res


def objective_rise(node, X, params, penalty):
    obj = 0
    for i in prange(len(X)):
        obj += llk(node, X[i], params)
    obj /= len(X)
    obj += penalty * np.linalg.norm(params, ord=1)
    return obj


def rise(X, penalty, delta, batch_size, max_gradient_steps=100):
    res = {}
    train_size, num_nodes = X.shape
    print("Number of training samples: {0}, number of nodes {1}".format(train_size, num_nodes))
    epochs = int(max_gradient_steps / (train_size / batch_size))
    print("Number of epochs : {}".format(epochs))
    params = np.random.normal(size=(num_nodes, num_nodes)) * 0.001
    loss = np.zeros((num_nodes, epochs))
    for epoch in tqdm(range(epochs)):
        old_params = np.copy(params)
        index_batch = np.random.choice(train_size, train_size, replace=False)
        dataloader = np.split(X[index_batch],
                              [batch_size*i for i in range(1, train_size // batch_size)])
        for node in range(num_nodes):
            loss[node][epoch] = objective_rise(node, X, params[node], penalty)
            for batch in dataloader:
                grad_ = grad(node, batch, params[node])
                params[node] = prox(params[node]-delta*grad_, penalty, delta)
        if np.allclose(old_params, params):
            break
    res["params"] = params
    res["loss"] = loss
    return res
