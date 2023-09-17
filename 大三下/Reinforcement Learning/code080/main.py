# -*- coding: utf-8 -*-
'''
@File    : main.py
@Time    : 2023/03/09 18:36:26
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 《强化学习》第四章-动态规划，例题4.2Jack租车问题
'''

import math
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import matplotlib.pyplot as plt

config = {  # matplotlib绘图配置
    "figure.figsize": (6, 6),  # 图像大小
    "font.size": 16, # 字号大小
    "font.sans-serif": ['SimHei'],   # 用黑体显示中文
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

class Environment:
    def __init__(self, out_lambda, in_lambda, action_sign) -> None:
        self.out_lambda, self.in_lambda = out_lambda, in_lambda
        self.action_sign = action_sign  # 用于判断action前的符号
    
    def step(self, state, action, in_num, out_num):
        out_num = min(out_num, state)
        new_state = max(min(state - out_num + action * self.action_sign + in_num, 20), 0)
        reward = 10 * out_num
        return int(new_state), reward

    def poisson(self, x, lamb):
        return np.power(lamb, x) / math.factorial(x) * np.exp(-lamb)
    
    def get_prob(self, in_num, out_num):
        prob = self.poisson(in_num, self.in_lambda) * self.poisson(out_num, self.out_lambda)
        return prob


class Policy_iterate:
    def __init__(self, T=10, sample_num=5000, max_storageA=20, max_storageB=20, gamma=0.9) -> None:
        self.T = T
        self.sample_num = sample_num
        self.max_storageA, self.max_storageB = max_storageA, max_storageB
        self.gamma = gamma
        self.min_delta_value = 1e-3  # 价值函数改变量阈值
        self.envA = Environment(out_lambda=3, in_lambda=3, action_sign=-1)
        self.envB = Environment(out_lambda=4, in_lambda=2, action_sign=1)
    
    def work(self):
        self.policy = np.zeros((self.max_storageA+1, self.max_storageB+1))
        self.value = np.zeros((self.max_storageA+1, self.max_storageB+1))
        for _ in range(self.T):
            self.plot_policy(_)
            self.policy_estimate()
            delta_value, policy_stable = self.policy_improve()
            if delta_value <= self.min_delta_value:
                print(f"第{_}次迭代，状态价值变换{delta_value:.3f}小于阈值{self.min_delta_value:.3f}，退出迭代")
                break
            if policy_stable is True:
                print(f"第{_}次迭代，策略已稳定，退出迭代")
                break

    def plot_policy(self, T):
        print(f"第{T}次迭代")
        print(self.policy)

        # 可视化策略
        plt.figure(figsize=(8, 5))
        x1, x2 = np.meshgrid(  # 创建离散网格点
            np.linspace(0, 20, 500),
            np.linspace(0, 20, 500)
        )
        # 计算高度值
        z = np.zeros_like(x1)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = self.policy[round(x2[i, j]), round(x1[i, j])]
        # 设置等高线划分点，会根据情况绘制等高线，若没有相应的数据点，则不会进行绘制
        levelz = range(-6, 6)
        plt.contourf(x1, x2, z, levels=levelz, cmap='tab20')  # 向由等高线划分的区域填充颜色
        plt.colorbar(label='行动')  # 制作右侧颜色柱，表示每种颜色对应的值

        plt.xlabel('B地汽车数目')
        plt.ylabel('A地汽车数目')
        plt.xticks(range(0, 21))
        plt.yticks(range(0, 21))
        plt.grid(False)
        plt.title(f"第{T}次迭代")
        plt.tight_layout()
        plt.savefig(f"policy{T}.png", dpi=300)

        # 可视化状态价值
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')
        x1, x2 = np.meshgrid(  # 创建离散网格点
            np.linspace(0, 20, 1000),
            np.linspace(0, 20, 1000)
        )
        z = np.zeros_like(x1)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = self.value[round(x2[i, j]), round(x1[i, j])]
        ax.plot_surface(x1, x2, z)
        ax.set_xlabel("B地汽车数目")
        ax.set_ylabel("A地汽车数目")
        plt.title(f"第{T}次迭代")
        plt.tight_layout()
        plt.savefig(f"value{T}.png", dpi=300)

        np.savetxt(f"policy{T}.txt", self.policy, fmt="%.0f")
        np.savetxt(f"value{T}.txt", self.value, fmt="%.2f")

    def calculate_value(self, stateA, stateB, action, value):
        new_value = 0
        for in_numA in range(6):
            for out_numA in range(6):
                probA = self.envA.get_prob(in_numA, out_numA)
                if probA < 0.0001:
                    continue
                for in_numB in range(6):
                    for out_numB in range(1, 7):
                        prob = probA * self.envB.get_prob(in_numB, out_numB)
                        new_stateA, rewardA = self.envA.step(stateA, action, in_numA, out_numA)
                        new_stateB, rewardB = self.envB.step(stateB, action, in_numB, out_numB)
                        reward = -4 * (new_stateA > 10) - 4 * (new_stateB > 10)  # 额外停车费
                        if action > 0:
                            reward = 10 * (rewardA + rewardB) - 2 * (action - 1)
                        else:
                            reward = 10 * (rewardA + rewardB) + 2 * action
                        new_value += prob * (reward + self.gamma * value[new_stateA, new_stateB])
        return new_value

    def policy_estimate(self):
        min_delta_value = 1e-3
        T = 100
        for _ in range(T):
            value_backup = self.value.copy()
            delta_value = 0
            for stateA in range(self.max_storageA+1):
                for stateB in range(self.max_storageB+1):
                    old_value = self.value[stateA, stateB]
                    action = self.policy[stateA, stateB]
                    new_value = self.calculate_value(stateA, stateB, action, value_backup)
                    self.value[stateA, stateB] = new_value
                    delta_value = max(delta_value, abs(new_value - old_value))
            print(delta_value)
            if delta_value <= min_delta_value:
                break

    def policy_improve(self):
        delta_value, policy_stable = 0, True
        for stateA in range(self.max_storageA+1):
            for stateB in range(self.max_storageB+1):
                old_action = self.policy[stateA, stateB]
                max_value = 0
                for action in range(-5, 6):
                    if (action < 0 and stateB < -action) or (action > 0 and stateA < action):
                        continue
                    new_value = self.calculate_value(stateA, stateB, action, self.value)
                    if new_value > max_value:
                        max_value = new_value
                        self.policy[stateA, stateB] = action
                if old_action != self.policy[stateA, stateB]:
                    policy_stable = False
                delta_value = max(delta_value, abs(self.value[stateA, stateB] - max_value))
        return delta_value, policy_stable

if __name__ == '__main__':
    np.random.seed(42)
    agent = Policy_iterate()
    agent.work()
