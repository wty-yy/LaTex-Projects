# -*- coding: utf-8 -*-

"""
@File    : hw5.py
@Author  : wtyyy
@Time    : 2022/10/16 20:34
"""

import numpy as np
import matplotlib.pyplot as plt

f = [0 for _ in range(4)]
f[0] = lambda x: 1-np.abs(x)
f[1] = lambda x: np.sqrt(1-x*x)
f[2] = lambda x: np.full_like(x, 1)
f[3] = lambda x: np.power(1-np.power(x,4), 1/4)
space = np.linspace(-1, 1, 500)
linestyle = ['-', '-.', '--', ':']
label = ['$||\cdot||_1$', '$||\cdot||_2$', '$||\cdot||_3$', '$||\cdot||_4$']

def copy(x, k=1):
    return np.concatenate([x, k * x[::-1], x[0:1]])

plt.figure(figsize=(5, 5))
for i in range(4):
    plt.plot(copy(space), copy(f[i](space), -1), ls=linestyle[i], label=label[i])
plt.legend()
plt.savefig('1.4.1.pdf')
plt.show()