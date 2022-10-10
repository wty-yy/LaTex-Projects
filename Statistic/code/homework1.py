# -*- coding: utf-8 -*-

"""
@File    : homework1.py
@Author  : wtyyy
@Time    : 2022/10/1 11:01
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

config = {
    "font.family": 'serif', # 衬线字体
    "figure.figsize": (10, 5),  # 图像大小
    "font.size": 14, # 字号大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # 渲染数学公式字体
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

def t_distribution(x, k):
    return math.gamma((k+1)/2) / (math.gamma(k/2) * math.sqrt(k*math.pi) * np.power(1+np.power(x,2)/k, (k+1)/2))

def normal_distribution(x, mu, sigma):
    return 1 / (math.sqrt(2*math.pi)*sigma) * np.exp(-(x-mu)*(x-mu) / (2*sigma*sigma))

def work1():
    def draw_mean(k, N, ls):
        T = 10000000
        mean = []
        for _ in tqdm(range(T)):  # 共计算T个样本
            x = np.random.chisquare(k, N)  # 随机出来一个chi分布的随机变量，每个样本有N个个体
            mean.append(x.mean())  # 求期望
        hist, bins = np.histogram(mean, bins=120)  # 绘制直方图
        y = hist * 1. / hist.sum()  # 归一化处理
        plt.plot(bins[:-1], y, label=f'k={k}, N={N}', ls=ls)
        # plt.hist(mean, bins=100, label=f'k={k}, N={N}')
    fig = plt.figure(figsize=(8, 4))
    draw_mean(10, 1, '-')
    draw_mean(10, 5, ':')
    draw_mean(10, 10, '-.')
    draw_mean(10, 100, '--')
    plt.legend()
    fig.tight_layout()
    plt.savefig('pg1.png', dpi=600)
    plt.show()

def work2():
    x = np.linspace(-6, 6, 5000)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, normal_distribution(x, 0, 1), label='$N(0,1)$')
    plt.plot(x, t_distribution(x, 1), label='$t(1)$', ls=':')
    plt.plot(x, t_distribution(x, 3), label='$t(3)$', ls='-.')
    plt.plot(x, t_distribution(x, 30), label='$t(30)$', ls='--')
    plt.plot(x, t_distribution(x, 100), label='$t(100)$', ls=':')
    plt.legend()
    fig.tight_layout()
    plt.savefig('pg2.png', dpi=600)
    plt.show()

def work3():
    mu = 0
    sigma = 1
    T = int(1e5)
    batch = int(1e5)
    moment, MLE = [], []
    for _ in tqdm(range(T)):
        x = np.random.uniform(mu-np.sqrt(3)*sigma, mu+np.sqrt(3)*sigma, batch)
        moment.append(x.mean())
        MLE.append((x.min() + x.max()) / 2)
    E = [mu-np.mean(moment), mu-np.mean(MLE)]
    std = [np.std(moment), np.std(MLE)]
    print(f'矩估计偏: {E[0]}, MLE偏: {E[1]}')
    print(f'矩估计方差: {std[0]}, MLE方差: {std[1]}')
    print(f'矩估计均方误差: {std[0]+E[0]*E[0]}, MLE均方误差: {std[1]+E[1]*E[1]}')


work1()
work2()
work3()
