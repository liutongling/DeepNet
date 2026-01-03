# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient

# 两层神经网络架构
class TwoLayerNet:
    # 参数得初始化，并且保存在字典中
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}
    # 正向传播函数，预测作用 回归一般用恒等函数，分类一般用softmax函数
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        # print("this is W1:")
        # print(W1)
        # print("this is b1:")
        # print(b1)
        # print("this is W2:")
        # print(W2)
        # print("this is b2:")
        # print(b2)
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        #print("z1")
        #print(z1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        #print(y.shape)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):
        loss_W = lambda W:self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

