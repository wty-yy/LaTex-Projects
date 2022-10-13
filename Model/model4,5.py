# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: model4,5.py
@time: 2022/10/9 13:58
"""
import numpy as np
import matplotlib.pyplot as plt

config = {
    "font.family": 'serif', # 衬线字体
    "figure.figsize": (6, 6),  # 图像大小
    "font.size": 14, # 字号大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # 渲染数学公式字体
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

k1 = 1
k2 = 0.5
A = 10
T = 1

def xn(t, x, y):
    return k1 * (y + A) / (k2 - k1) * (np.exp(-k1*t)-np.exp(-k2*t))+x*np.exp(-k2*t)

def yn(t, n):
    if n == 0:
        return np.zeros_like(t)
    return A * np.exp(k1*(T-t)) * (1-np.exp(-n*k1*T)) / (np.exp(k1*T) - 1)

def ynn(t, y):
    return (y + A) * np.exp(-k1*t)

t_space = np.linspace(0, T, 100)
last_x, last_y = 0, 0
X = []
Y = []
t_range = []
for i in range(30):
    t0 = T * i
    t = t_space + t0
    x = xn(t_space, last_x, yn(T, i))
    last_x = x[-1]
    # y = ynn(t_space, last_y)
    y = yn(t_space, i+1)
    last_y = y[-1]
    # y = yn(t_space, i)
    X += x.tolist()
    Y += y.tolist()
    t_range += t.tolist()
plt.plot(t_range, X, label='药物浓度$x$')
plt.plot(t_range, Y, label='残量$y$')
plt.legend()
plt.title(f'分解速率{k1},吸收速率{k2},周期摄入量{A},周期{T}')
plt.xlabel('时间')
plt.savefig('浓度与残量关于时间变化关系.png', dpi=600)
plt.show()
print(f'{last_x=}, {last_y=}, {np.max(X)=}')
pred_x = k1 * A * (np.exp(k2*T)-np.exp(k1*T)) / (k2-k1) / (np.exp(k1*T)-1) / (np.exp(k2*T)-1)
pred_y = A/ (np.exp(k1 * T) - 1)
tmp1 = k1 / (k2-k1)
tmp2 = k2 / (k2-k1)
pred_max = k1*A/(k2-k1)*(np.power(k1/k2, tmp1)-np.power(k1/k2, tmp2))*np.power(np.exp(k2*T)-1, tmp1)/np.power(np.exp(k1*T)-1, tmp2)
print(f'{pred_x=}, {pred_y=}, {pred_max=}')
