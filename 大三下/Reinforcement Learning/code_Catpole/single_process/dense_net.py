# -*- coding: utf-8 -*-
'''
@File    : dense_net.py
@Time    : 2023/04/17 10:45:34
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 策略网络为全连接神经网络
    Input: 4
    Dense: 32
    Dense: 32
    Output: 2
    activate function: ReLU
    Loss function:MSE
    optimizer: SGD
'''

import tensorflow as tf
import numpy as np
import random
from pathlib import Path

BATCH_SIZE = 32
MEMORY_SIZE = 10000

EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

CHECKPOINT_PREFIX = Path(r"./checkpoints")

class DenseModel:
    def __init__(self, learning_rate=0.001, gamma=0.95, epsilon=EPSILON_MAX, policy='Q_learning') -> None:
        self.lr = learning_rate
        self.gamma = gamma
        self.policy = policy
        self.epsilon = epsilon

        self.model = self.dense_net()
        self.model.build(input_shape=(None, 4))
        # self.model.summary()

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)
        self.loss = tf.keras.losses.MSE
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        self.memory = [0 for _ in range(MEMORY_SIZE)]
        self.used = np.zeros(MEMORY_SIZE)
        self.n_samples = 0
    
    def dense_net(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", name="Dense1"),
            tf.keras.layers.Dense(32, activation="relu", name="Dense2"),
            tf.keras.layers.Dense(2, name="Output"),
        ])
        return model
    
    def act(self, state, use_epsilon=True):
        sample = np.random.rand()
        if sample <= self.epsilon and use_epsilon:
            return np.random.randint(0, 2)
        else:
            return np.argmax(self.predict(state)[0])

    def remember(self, state, action, reward, state_, terminal):
        sars = (state, action, reward, state_, terminal)
        self.memory[self.n_samples % MEMORY_SIZE] = sars
        self.used[self.n_samples % MEMORY_SIZE] = 0
        self.n_samples += 1

    def sample_memory(self):
        size = min(MEMORY_SIZE, self.n_samples)
        p = 1 / (1 + self.used[:size])
        p = p / np.sum(p)
        idx_sample = np.random.choice(range(size), size=BATCH_SIZE, replace=False, p=p)
        self.used[idx_sample] += 1
        return [self.memory[x] for x in idx_sample]

    def fit(self):
        if self.n_samples < BATCH_SIZE: return
        # batch = np.array(random.sample(self.memory, BATCH_SIZE), dtype=object)
        batch = np.array(self.sample_memory(), dtype=object)
        batch = np.c_[batch, self.predict(np.concatenate(batch[:, 0])), self.predict(np.concatenate(batch[:, 3]))]
        def make_y(a):
            action = a[1]
            target = a[2]  # reward
            if not a[4]:  # terminal
                q_eval = a[7:9]
                policy_distrib = q_eval / (np.sum(q_eval) + 0.001)
                if self.policy == 'Q_learning':
                    target += self.gamma * np.max(q_eval)
                elif self.policy == 'E_sarsa':
                    target += self.gamma * policy_distrib * q_eval
            y = a[5:7]
            y[action] = target
            return target
        q_values = np.apply_along_axis(make_y, 1, batch)
        x = np.concatenate(batch[:, 0]).reshape(-1, 4)
        y = batch[:, 5:7].astype(np.float32)
        self.model.fit(x, y, batch_size=4, verbose=0)
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)
        return np.mean(q_values)

    def predict(self, x):
        x = x.reshape(-1, 4)
        return self.model.predict(x, verbose=0)
    
    def save_weights(self, fname=""):
        self.model.save_weights(CHECKPOINT_PREFIX.joinpath("dqn" + fname))
    
    def load_weights(self, path):
        self.model.load_weights(path).expect_partial()
        config = self.model.optimizer.get_config()
        self.model.optimizer.from_config(config)

    def reset(self):
        self.epsilon = EPSILON_MAX
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.memory = [0 for _ in range(MEMORY_SIZE)]
        self.used = np.zeros(MEMORY_SIZE)
        self.n_samples = 0

if __name__ == '__main__':
    pass
