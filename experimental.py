import numpy as np
def function(x):
    return np.sum(x**2)

# 计算梯度的方法
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        x[i] += h
        fxh1 = f(x)
        x[i] -= 2*h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2)/(2*h)
        x[i] += h
    print(grad)
    return grad
# 对sigmod函数的求导
class Sigmod:
    def __init__(self):
        self.a = 0
    def forward1(self,x):
        return -1*x
    def forward2(self,x):
        return np.exp(-x)

    def forward3(self,x):
        return 1+np.exp(-x)

    def forward4(self,x):
        self.a = 1 / (1 + np.exp(-x))
        return self.a
    def div1(self,dout):
        return -(1/(self.forward3(self.a)**2))

    def add2(self,dout):
        return dout
    def exp3(self,dout):
        return dout*self.forward2(self.a)

    def mul4(self,dout):
        return -dout

    def backward(self,dout):
        d1 = self.div1(dout)
        print(d1)

        d2 = self.add2(d1)
        print(d2)
        d3 = self.exp3(d2)
        print(d3)
        d4 = self.mul4(d3)
        print(d4)
        return d4
