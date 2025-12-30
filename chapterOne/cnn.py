# coding: utf-8
import sys,os
sys.path.append(os.pardir) # 设定当前的路径
import numpy as np
from dataset.mnist import load_mnist

import matplotlib.pyplot as plt
from two_layer_net import *