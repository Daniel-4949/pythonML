import numpy as np
import matplotlib.pyplot as plt

# 梯度下降法Gradient Descent, GD

def func(x):
    return np.square(x) #y = x^2
def dfunc(x):
    return 2*x #y' = 2x

def GD(x_start, df, epochs, lr):    
    #   梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
    #     :param x_start: x的起始點    
    #     :param df: 目標函數的一階導函數    
    #     :param epochs: 反覆運算週期    
    #     :param lr: 學習率    
    #     :return: x在每次反覆運算後的位置（包括起始點），長度為epochs+1    
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        # v表示x要改變的幅度  
        v = -dx * lr  
        x += v
        xs[i+1] = x
    return xs
