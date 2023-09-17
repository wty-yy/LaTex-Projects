# -*- coding: utf-8 -*-
'''
@File    : main.py
@Time    : 2023/04/17 08:04:40
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 用网络拟合Q函数解决gymnasium库中的cartpole平衡问题
'''

import numpy as np
import gymnasium as gym
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from generate_gif import save_frames_as_gif
import template.reinforcement as RL
from dense_net import DenseModel, EPSILON_MAX
from test_demo.clock import Clock

PATH_FIGURES = Path.cwd().joinpath("figures/Q_learning")
PATH_FIGURES.mkdir(parents=True, exist_ok=True)

class Agent(RL.Agent):
    best = {
        'step': 0,
        'frames': [],
        'episode': 0,
    }
    history = {
        'step': [],
        'loss': [],
        'q_value': [],
    }
    def __init__(self, env_name="CartPole-v1", policy='Q_learning', alpha=0.001, epsilon=EPSILON_MAX, gamma=0.95, seed=42, episodes=60, epochs=1, name="") -> None:
        """
            policy: ['Q_learning', 'E_sarsa']
        """
        super().__init__(env_name, alpha, epsilon, gamma, seed, episodes, epochs)
        assert(policy in ['Q_learning', 'E_sarsa'])
        self.policy = policy
        self.model = DenseModel(learning_rate=alpha, gamma=gamma, policy=policy)
        self.name = name
    
    def train(self, verbose=False):
        env = gym.make(self.env_name, render_mode="rgb_array")
        for episode in tqdm(range(self.episodes)):
            frames = []
            episodes = []
            q_values = []
            state, _ = env.reset()
            for step in range(501):
                if verbose:
                    frames.append(env.render())
                action = self.model.act(state)
                state_, reward, terminal, _, _ = env.step(action)
                if terminal and step != 500:
                    reward = -10
                episodes.append((state, action, reward))
                self.model.remember(state, action, reward, state_, terminal)
                q_value = self.model.fit()
                if q_value is not None: q_values.append(q_value)
                state = state_
                if terminal: break
            self.history['step'].append(step)
            if step > self.best['step']:
                self.best = {'step': step, 'frames': frames, 'episode': episode}
                self.model.save_weights(fname='_best')
                if verbose:
                    save_frames_as_gif(frames=agent.best['frames'], fname=self.name, path=PATH_FIGURES, step=step)
            self.history['q_value'].append(np.mean(q_values))
            self.save_plot_figures()
            self.model.save_weights()
        env.close()
    
    def evaluate(self, verbose=True, disable_tqdm=False):
        env = gym.make(self.env_name, render_mode="rgb_array")
        for episode in tqdm(range(self.episodes), disable=disable_tqdm):
            frames = []
            episodes = []
            q_values = []
            state, _ = env.reset()
            for step in range(501):
                if verbose:
                    frames.append(env.render())
                action = self.model.act(state, use_epsilon=False)
                state_, reward, terminal, _, _ = env.step(action)
                if terminal and step != 500:
                    reward = -10
                episodes.append((state, action, reward))
                q_value = np.mean(self.model.predict(state))
                if q_value is not None: q_values.append(q_value)
                state = state_
                if terminal: break
            self.history['step'].append(step)
            if step > self.best['step']:
                self.best = {'step': step, 'frames': frames, 'episode': episode}
                if verbose:
                    save_frames_as_gif(frames=self.best['frames'], fname=self.name, path=PATH_FIGURES, step=step)
            self.history['q_value'].append(np.mean(q_values))
            # self.save_plot_figures()
        env.close()

    def multi_evaluate(self, items):
        clock = Clock()
        idx, base_agent = items
        print(idx, base_agent.name)
        # agent = deepcopy(base_agent)
        # agent.name = "evaluation" + str(idx)
        # print(f"{idx} {agent.name} {id(agent.model)}")
        # agent.evaluate(verbose=True, disable_tqdm=True)
        clock.end(f"{idx} ")

    def save_plot_figures(self):
        best_episode = np.argmax(self.history['step'])
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"scores max:{np.max(self.history['step'])}")
        plt.plot(self.history['step'], label='Q_learning')
        plt.plot(best_episode, self.history['step'][best_episode], '.r', label='best step')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title('Mean of Q Values')
        plt.plot(self.history['q_value'], label='Q_learning')
        plt.legend()
        plt.tight_layout()
        plt.savefig(PATH_FIGURES.joinpath(f"training_scores_loss_{self.name}.png"), dpi=100)
        plt.close()
    
    def reset(self):
        self.model.reset()
        self.history = {
            'step': [],
            'loss': [],
            'q_value': [],
        }
    
    def __deepcopy__(self, momo):
        new_obj = Agent()
        new_obj.__dict__.update(self.__dict__)
        return new_obj

if __name__ == '__main__':
    agent = Agent(policy='Q_learning', alpha=1e-3, episodes=60)
    for i in range(10):
        print(f"Restart {i} times...")
        agent.name = str(i)
        agent.reset()
        agent.train(verbose=False)
        print("best step:", agent.best['step'], "at episode", agent.best['episode'])
        # save_frames_as_gif(frames=agent.best['frames'], fname='Q_learning', path=PATH_FIGURES)
