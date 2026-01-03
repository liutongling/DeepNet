# coding: utf-8
import numpy as np


def identity_function(x):
    pass


def step_function(x):
    pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    pass


def relu(x):
    pass


def relu_grad(x):
    pass


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def mean_squared_error(y, t):
    pass


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    pass
