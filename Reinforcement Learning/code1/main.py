# -*- coding: utf-8 -*-
'''
@File    : main.py
@Time    : 2023/03/05 20:19:02
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 强化学习中文版第33面，练习2.5，10臂赌博机比较均值估计和指数近因加权估计两种方法，
           均采用epsilon=0.1贪心选择动作，执行10000步.
'''

from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

config = {  # matplotlib绘图配置
    "figure.figsize": (6, 6),  # 图像大小
    "font.size": 16, # 字号大小
    "font.sans-serif": ['SimHei'],   # 用黑体显示中文
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

class Bandit:  # 赌博机类
    def __init__(self, n, init_means=0) -> None:
        # init_means: 赌博机初始均值
        self.n = n  # 赌博机臂个数
        self.means = np.full(self.n, init_means, dtype='float')  # 均值初始值

    def move_state(self):
        return np.random.normal(0, 0.01, self.n)  # 随机游走变换函数

    def step(self, action):  # 执行一个action，并返回其收益
        assert(0 <= action < self.n)
        self.means += self.move_state()
        mean_now = self.means[action]
        rate = np.sum(self.means <= mean_now) / self.n
        return np.random.normal(self.means[action], size=1)[0], rate

def train(strategy, alpha=0.1):
    # 两种策略 average 和 recency
    bandit = Bandit(n)
    N = np.zeros(n)
    Q = np.zeros(n)
    history = [[], []]   # 存储每个时刻的收益和最优动作占比
    for t in range(T):
        if np.random.random(1)[0] < epsilon:  # 随机选取一个动作
            action = np.random.randint(0, n)
        else:  # 贪心选择最大价值的动作
            action = np.argmax(Q)
        reward, rate = bandit.step(action)
        delta = reward - Q[action]
        N[action] += 1
        Q[action] +=  delta / N[action] if strategy == 'average' else delta * alpha
        history[0].append(reward)
        history[1].append(rate)
    return np.array(history)

def plot_reward_rate(strategy, m=2000):  # m为实验次数
    history = np.zeros((2, T))
    for _ in tqdm(range(m)):
        history += train(strategy=strategy)
    history /= m
    ax[0].plot(range(T), history[0], label='均值估计法' if strategy=='average' else '指数近因估计法')
    ax[1].plot(range(T), history[1], label='均值估计法' if strategy=='average' else '指数近因估计法')
    

if __name__ == '__main__':
    n, T = 10, 10000
    epsilon = 0.1
    np.random.seed(42)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    plot_reward_rate('average', m=2000)
    plot_reward_rate('recency', m=2000)

    ax[0].set_ylabel('平均收益')
    ax[1].set_ylabel('最优动作/%')
    ax[1].set_xlabel('训练步数')
    ax[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax[0].legend()
    ax[1].legend()
    plt.savefig('33页练习2.5.png', dpi=300)
    plt.show()