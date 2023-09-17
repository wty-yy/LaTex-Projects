# -*- coding: utf-8 -*-
'''
@File    : player.py
@Time    : 2023/04/20 16:38:25
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : Player实现了多线程与环境交互，每个线程就是一个Player：一个在一个独立环境中游玩的玩家
由Agent根据每个Player的state，向每个Player分别发送action指令，Player执行action之后，
会将环境的新state_以 (state, action, reward, state_, terminal, init, step) 形式返回

- N_PLAYER(int): Player总数，子程序数目
- Player(class): Player类，使用MyManger.Player()进行创建，由player_act函数进行主线程与子线程之间的交互
- player_action((player, action)):
使用pool.map(func=player_action, iterable=[(player1, action1), (player2, action2),..., (playerN, actionN)])
开始多线程，例如：
with Pool() as pool:
    results = list(pool.map(func=player_act, iterable=zip(players, actions)))
- LogManager(class): 
两个路径变量（会自动创建文件夹）
  1. path_figures: 图片存储目录
  2. path_gif: GIF动图存储目录

主要有三个函数，2,3需要先执行1对日志信息进行更新后使用
  1. update(players): 将player列表传入，读取player内部的logs
  2. plot_logs(): 以多组图形式绘制每个player的step图像，多组图的列数由PLOT_COLUMNS_OF_FIGURES控制
  3. plot_best_frames(): 将每个player的最优一幕的结果绘制成gif文件，目录为log_manager.path_gif

具体使用方法可参考下面的main()函数
'''

from multiprocessing import Process, Pool
from multiprocessing.managers import BaseManager
import gymnasium as gym
from tools.clock import Clock
from tools.mytqdm import MyTqdm
from tools.generate_gif import save_frames_as_gif
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import constant as const

PATH_FIGURES = Path(__file__).parent.joinpath("test_figures")
PATH_FIGURES.mkdir(parents=True, exist_ok=True)
PATH_GIF = PATH_FIGURES.joinpath("gif")
PATH_GIF.mkdir(parents=True, exist_ok=True)
N_PLAYER = const.N_PLAYER
EPISODES = 100
PLOT_COLUMNS_OF_FIGURES = 2

class Player:
    def __init__(self, env_name="CartPole-v1", name="env", is_render=True) -> None:
        self.name = str(name)
        self.is_render = is_render
        self.env = gym.make(env_name, render_mode="rgb_array")
        # self.env = gym.make(env_name, render_mode="human")
        self.state = np.nan
        self.terminal = True
        self.step = 0
        self.frames = []
        self.logs = {
            'steps': [],
            'best': {
                'step': 0,
                'frames': [],
            }
        }
    
    def add_logs(self):
        if self.step != 0:
            self.logs['steps'].append(self.step)
            if self.step > self.logs['best']['step']:
                self.logs['best']['step'] = self.step
                self.logs['best']['frames'] = self.frames
        self.step = 0
        self.frames = []
    
    def get_logs(self):
        return self.logs
    
    def get_name(self):
        return self.name
    
    def act(self, action):
        state, reward, init = self.state, np.nan, False
        if self.terminal:
            self.add_logs()
            state_, _ = self.env.reset(seed=np.random.randint(1,100))
            terminal = False
            init = True
        else:
            if self.is_render:
                self.frames.append(self.env.render())
            self.step += 1
            state_, reward, terminal, _, _ = self.env.step(action)
            if terminal and self.step != 500:
                reward = -10
            if self.step == 500:
                terminal = True
        self.state, self.terminal = state_, terminal
        return (state, action, reward, state_, terminal, init, self.step)

class MyManager(BaseManager): pass
MyManager.register('Player', Player)

def player_act(items):
    player, action = items
    return player.act(action)

# def plot_player_logs(player, ax:plt.Axes):
#     logs = player.get_logs()
#     name = player.get_name()
#     ax.set_title(f"player: {name}")
#     if len(logs) == 0: return
#     ax.plot(logs['steps'])
# 
# def plot_update_player_logs(players):
#     fig, axes = plt.subplots(N_PLAYER, 1, figsize=(N_PLAYER*3, 6))
#     for idx, player in enumerate(players):
#         plot_player_logs(player=player, ax=axes[idx])
#     fig.tight_layout()
#     fig.savefig(PATH_FIGURES.joinpath(f"player_logs.png"), dpi=100)
#     plt.close()

class LogManager:
    def __init__(self, n_player=N_PLAYER, path_figures=PATH_FIGURES, path_gif=PATH_GIF) -> None:
        self.n_player = n_player
        self.logs = [{} for _ in range(self.n_player)]
        self.path_figures = path_figures
        self.path_gif = path_gif
        self.path_figures.mkdir(parents=True, exist_ok=True)
        self.path_gif.mkdir(parents=True, exist_ok=True)
    
    # def reset(self):
    #     self.steps = np.zeros(N_PLAYER)
    #     self.logs = [[] for _ in range(N_PLAYER)]

    # def update(self, terminals):
    #     self.steps += 1
    #     for idx, terminal in enumerate(terminals):
    #         if terminal:
    #             self.logs[idx].append(self.steps[idx])
    #             self.steps[idx] = 0

    def update(self, players):
        for idx, player in enumerate(players):
            self.logs[idx] = player.get_logs()
    
    def plot_logs(self):
        n_c = int(min(self.n_player, PLOT_COLUMNS_OF_FIGURES))
        n_r = int((self.n_player - 1) / PLOT_COLUMNS_OF_FIGURES + 1)
        fig, axes = plt.subplots(n_r, n_c, figsize=(6 * n_c, 3 * n_r))
        axes = axes.reshape(-1)
        for idx, log in enumerate(self.logs):
            ax = axes[idx]
            ax.set_title(f"player: {idx}")
            if len(log['steps']) == 0: continue
            ax.plot(log['steps'], '.-')
            best_step = np.max(log['steps'])
            best_episode = np.argmax(log['steps'])
            ax.plot(best_episode, best_step, 'r*')
            ax.plot([0, best_episode], [best_step, best_step], 'g--')
            if len(log['steps']) > 1:
                ax.set_xlim(0, len(log['steps'])-1)
        fig.tight_layout()
        fig.savefig(self.path_figures.joinpath(f"player_logs.png"), dpi=300)
        plt.close()

    def multi_save_frames_as_gif(self, items):
        frames, name, path = items
        if len(frames) == 0: return
        save_frames_as_gif(*items)
        np.save(const.PATH_LOGS.joinpath(f"frames{name}"), frames)
        print(f"Complete plot {name}.gif")

    def plot_best_frames(self):
        input_args = []
        for idx, log in enumerate(self.logs):
            frames = log['best']['frames']
            input_args.append((frames, f"player{idx}", self.path_gif))

        for items in input_args:
            self.multi_save_frames_as_gif(items)
        # with Pool() as pool:
        #     list(pool.imap_unordered(func=self.multi_save_frames_as_gif, iterable=input_args))

def main():
    clock = Clock()
    with MyManager() as manager:
        episodes = EPISODES
        log_manager = LogManager()
        players = [manager.Player(name=name) for name in range(N_PLAYER)]
        with MyTqdm(total=EPISODES) as pbar:
            while episodes > 0:
                actions = np.random.randint(0, 2, size=N_PLAYER)
                with Pool() as pool:
                    results = list(pool.map(func=player_act, iterable=zip(players, actions)))
                    results = np.array(results, dtype=object)
                    terminals = results[:, 4]
                    episodes -= np.sum(terminals)
                    pbar.update(np.sum(terminals))
        # for _, player in enumerate(players):
        #     print(player.get_name())
        #     print(player.get_logs()['steps'])
        log_manager.update(players)
        log_manager.plot_logs()
        log_manager.plot_best_frames()
        print(f"Total episodes: {EPISODES - episodes}")
    clock.end()

if __name__ == '__main__':
    main()