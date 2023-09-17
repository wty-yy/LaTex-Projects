# -*- coding: utf-8 -*-
'''
@File    : templete.py
@Time    : 2023/04/03 15:42:20
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 保存Python中的各种类的模板
'''

import numpy as np

class Environment:
    state, terminal = None, None
    def __init__(self, seed=42) -> None:
        """
        Input:
            seed: 随机种子
        """
        self.seed = seed

    def step(self, action):
        """
        Input:
            action:   由智能体输入动作，环境返回状态
        Output:
            reward:   执行动作后的奖励
            state:    下一个状态
            terminal: 是否到达终止状态
        """
        pass

    def reset(self):
        """
            重置环境状态
        """
        pass

class Agent:
    def __init__(self, env_name, alpha=0.001, epsilon=0.05, gamma=0.9, seed=42, episodes=500, epochs=1) -> None:
        """
        Input:
            env_name:     环境名
            alpha:   学习率
            epsilon: epsilon-贪心
            gamma:   折扣回报中的折扣率
            seed:    随机种子
            episode: 训练的总回合数，环境重启次数
            epoch:   总共训练次数，智能体重启次数，用于均值绘制曲线
        """
        self.env_name = env_name
        self.alpha, self.epsilon, self.gamma, self.seed, self.episodes, self.epochs = \
            alpha, epsilon, gamma, seed, episodes, epochs
        # np.random.seed(self.seed)
        
    def start(self):
        """
            开始设置策略，记录策略返回值，并绘制图像
        """
        pass

    def train(self):
        """
        智能体根据初始化定义的参数进行学习
        Input:
            policy:  智能体遵循的学习策略
        Output:
            history: 返回学习过程中的历史数据，用于绘图
        """
        pass

if __name__ == '__main__':
    pass
