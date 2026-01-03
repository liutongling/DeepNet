# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    pass

def numerical_gradient_2d(f, X):
    pass


# 用导数定义实现梯度
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)# f(x+h)
        #print("this is fxh1")
        #print(fxh1)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad