# -*- coding: utf-8 -*-

"""
@File    : problem5.8.py
@Author  : wtyyy
@Time    : 2022/10/23 13:07
"""

import numpy as np
import pandas as pd

# 绘制表格参数
cols = ['A正面期望', 'A背面期望', 'B正面期望', 'B背面期望', 'A正面概率', 'B正面概率']
df = pd.DataFrame(columns=cols)  # 绘制表格

T = 10  # 总迭代次数
prA, prB = 0.3, 0.7  # 硬币A,B正面朝上的概率
samples = [4, 6, 0, 9, 5]  # 每个样本中正面朝上的个数
for _ in range(T):
    expectA, expectB = np.zeros(2), np.zeros(2)  # 硬币A,B的期望
    for i in range(len(samples)):
        tmp1 = np.power(prA, samples[i]) * np.power(1 - prA, 10 - samples[i])
        tmp2 = np.power(prB, samples[i]) * np.power(1 - prB, 10 - samples[i])
        chooseA = tmp1 / (tmp1 + tmp2)  # 选择硬币A的概率
        chooseB = 1 - chooseA  # 选择硬币B的概率
        expectA += np.array([samples[i] * chooseA, (10 - samples[i]) * chooseA])
        expectB += np.array([samples[i] * chooseB, (10 - samples[i]) * chooseB])
    prA = expectA[0] / np.sum(expectA)
    prB = expectB[0] / np.sum(expectB)
    tmp = pd.DataFrame(
        np.concatenate((expectA, expectB, np.array([prA]), np.array([prB]))).reshape([1, -1]),
        columns=cols)
    df = pd.concat([df, tmp])
df = df.reset_index(drop=True)
df.index += 1
df.index.name = '迭代次数'
print(df)
df.round(2).to_excel('ans8.xlsx')