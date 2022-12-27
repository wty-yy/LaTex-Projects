"""
@Author: wty-yy
@Date: 2022-12-27 18:10:00
@LastEditTime: 2022-12-27 20:01:15
@Description: 
"""

import numpy as np
n = np.array([[442, 514], [38, 6]])
N = np.sum(n)
p = np.zeros_like(n, np.float32)
for i in range(2):
    for j in range(2):
        p[i,j] = n[i,j] / N
print(p)
l = 1
q = 0
# for i in range(2):
#     l *= np.power(sum(p[i,:]), sum(n[i,:]))
#     l *= np.power(sum(p[:,i]), sum(n[:,i]))
for i in range(2):
    for j in range(2):
        if i == 0:
            l *= np.power(sum(p[j,:]), sum(n[j,:]))
        else:
            l *= np.power(sum(p[:,j]), sum(n[:,j]))
        l /= np.power(p[i,j], n[i,j])
        q += np.power(n[i,j] - N * np.sum(p[i,:]) * np.sum(p[:,j]), 2) / (N * np.sum(p[i,:]) * np.sum(p[:,j]))
print(l, -2*np.log(l), q)