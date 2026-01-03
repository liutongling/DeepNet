# coding: utf-8
import sys,os
sys.path.append(os.pardir) # 设定当前的路径
import numpy as np
from dataset.mnist import load_mnist

import matplotlib.pyplot as plt
from two_layer_net import *

if __name__ == '__main__':
    #print("hello world")
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    # 超参数
    iter_num = 5000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1 #梯度进的步骤
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]
        #print(x_batch)
        #print(x_batch.shape)

        grad = network.gradient(x_batch, y_batch)

        for param in network.params.keys():
            network.params[param] -= learning_rate * grad[param]

        loss = network.loss(x_batch, y_batch)
        train_loss_list.append(loss)
        print("this is loss%f",loss)

