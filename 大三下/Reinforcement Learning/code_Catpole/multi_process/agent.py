# -*- coding: utf-8 -*-
'''
@File    : agent.py
@Time    : 2023/04/20 17:22:12
@Author  : wty-yy
@Version : 1.1
@Blog    : https://wty-yy.space/
@Desc    : 智能体:
- 脑子: 深度神经网络 DNN
- 记忆: memory
- 行为:
    - train(): 从记忆中采样BATCH_SIZE*player.N_PLAYER个数据，对DNN进行训练
    - act(state, is_epsilon):
        根据环境状态states，给出最优行动/epsilon-决策
    - save_weights(): 将DNN权重保存到PATH_CHECKPOINT+agent.name文件中
    - load_weights(path): 从path中读取DNN权重

v1.1: 优化fit内函数写法，使输入维度更为一般化
'''

import tensorflow as tf
import numpy as np
import player
import swapper
import matplotlib.pyplot as plt
from pathlib import Path
import constant as const

LEARNING_RATE = const.LEARNING_RATE
GAMMA = 0.95
BATCH_SIZE = const.BATCH_SIZE
MEMORY_SIZE = 10000
STATE_DIM = 4

EPSILON_MAX = const.EPSILON_MAX
EPSILON_MIN = const.EPSILON_MIN
EPSILON_DECAY = const.EPSILON_DECAY

PATH_CHECKPOINT = const.PATH_CHECKPOINT

class Agent:
    def __init__(self, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON_MAX, policy='Q_learning') -> None:
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.train_size = BATCH_SIZE * player.N_PLAYER

        self.model = self._dense_net()
        self.model.build(input_shape=(None, STATE_DIM))

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)
        self.loss = tf.keras.losses.MSE
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        self.memory = [0 for _ in range(MEMORY_SIZE)]
        self.used = np.zeros(MEMORY_SIZE)
        self.n_samples = 0

        self.logs = {
            "q_values": [],
        }

    def _dense_net(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", name="Dense1"),
            tf.keras.layers.Dense(32, activation="relu", name="Dense2"),
            tf.keras.layers.Dense(2, name="Output"),
        ])
        return model

    def act(self, state, is_epsilon=True):
        sample = np.random.rand()
        if sample <= self.epsilon and is_epsilon:
            return np.random.randint(0, 2, size=player.N_PLAYER)
        else:
            return np.argmax(self.predict(state), axis=1)

    def predict(self, x:np.ndarray):
        x = x.reshape(-1, 4)
        return self.model.predict(x, verbose=0)

    def remember(self, results):
        for sars in results:
            # sars=(state, action, reward, state_, terminal, init, step)
            if sars[5]: continue
            self.memory[self.n_samples % MEMORY_SIZE] = sars[:5]
            self.used[self.n_samples % MEMORY_SIZE] = 0
            self.n_samples += 1

    def _sample_memory(self):
        size = min(MEMORY_SIZE, self.n_samples)
        p = 1 / (1 + self.used[:size])
        p = p / np.sum(p)
        idx_sample = np.random.choice(range(size), size=self.train_size, replace=False, p=p)
        self.used[idx_sample] += 1
        return [self.memory[x] for x in idx_sample]

    def fit(self):
        if self.n_samples < self.train_size: return
        batch = self._sample_memory()
        # batch的一行: (state, action, reward, state_, terminal)
        states = np.array([x[0] for x in batch])
        states_ = np.array([x[3] for x in batch])
        preds = self.predict(states)
        preds_ = self.predict(states_)
        q_values = []
        for i in range(len(batch)):
            action = batch[i][1]
            target = batch[i][2]  # init from reward
            if not batch[i][4]:  # not terminal
                q_eval = preds_[i]
                policy_distrib = q_eval / (np.sum(q_eval) + 0.001)
                if self.policy == 'Q_learning':
                    target += self.gamma * np.max(q_eval)
                elif self.policy == 'E_sarsa':
                    target += self.gamma * np.dot(policy_distrib, q_eval)
            preds[i][action] = target
            q_values.append(target)
        x = states
        y = preds
        self.model.fit(x, y, batch_size=BATCH_SIZE, verbose=0)
        self.epsilon = max(self.epsilon * (EPSILON_DECAY ** player.N_PLAYER), EPSILON_MIN)
        # self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)
        self.logs['q_values'].append(np.mean(q_values))

    def plot_logs(self, fname='Q_values', path=const.PATH_FIGURES):
        fig, ax = plt.subplots()
        ax.plot(self.logs['q_values'])
        ax.set_title("Q Values")
        fig.savefig(path.joinpath(fname + '.png'), dpi=300)
        plt.close()

    def save_weights(self, fname=""):
        self.model.save_weights(PATH_CHECKPOINT.joinpath("dqn" + fname))
    
    def load_weights(self, path):
        self.model.load_weights(path).expect_partial()
        config = self.model.optimizer.get_config()
        self.model.optimizer.from_config(config)

if __name__ == '__main__':
    pass
