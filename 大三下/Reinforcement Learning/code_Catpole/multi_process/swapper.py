# -*- coding: utf-8 -*-
'''
@File    : main.py
@Time    : 2023/04/20 16:38:10
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : Agent与Player之间的交换器swapper，作为主线程将子线程Player与Agent的数据传输，并启动Agent的训练
'''
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tools.mytqdm import MyTqdm
from pathlib import Path
import player
import agent
import constant as const

class StandardManager:
    def __init__(self) -> None:
        self.upper_standard = const.N_PLAYER
        self.standard_step = const.STANDARD_STEP
        self.count = 0

    def update(self, results:np.ndarray):
        terminals = results[:, 4]
        steps = results[:, 6]
        self.count += np.sum(np.dot(terminals, steps) > self.standard_step)
    
    def check(self):
        return self.count < self.upper_standard

class Swapper:
    def __init__(self, episodes=100, path_figures=const.PATH_FIGURES) -> None:
        self.path_figures = Path(path_figures)
        self.path_figures.mkdir(parents=True, exist_ok=True)
        self.episodes = episodes
        self.agent = agent.Agent()
        self.best = {
            'step': 0
        }
    
    def start(self, is_train=False, is_render=True):
        with player.MyManager() as manager:
            episodes = self.episodes
            std_manager = StandardManager()
            log_manager = player.LogManager(path_figures=self.path_figures,
                                            path_gif=self.path_figures.joinpath("gif"))
            players = [manager.Player(name=name, is_render=is_render) for name in range(player.N_PLAYER)]
            states = None
            with MyTqdm(total=episodes) as pbar:
                with Pool(processes=const.N_PLAYER) as pool:
                    while episodes > 0 and std_manager.check():
                        if states is None:
                            actions = [0 for _ in range(player.N_PLAYER)]
                        else:
                            actions = self.agent.act(states, is_epsilon=is_train)
                        results = list(pool.map(func=player.player_act,
                                                iterable=zip(players, actions)))
                        results = np.array(results, dtype=object)
                        # (state, action, reward, state_, terminal, init, step)
                        states = np.concatenate(results[:, 3]).reshape(-1, 4)
                        mx_step = np.max(results[:, 6])
                        std_manager.update(results)
                        if mx_step > self.best['step']:
                            self.best['step'] = mx_step
                            self.agent.save_weights("_best")
                        terminals = results[:, 4]
                        episodes -= np.sum(terminals)
                        pbar.update(np.sum(terminals))
                        if is_train:
                            self.agent.remember(results)
                            self.agent.fit()
                        if np.sum(terminals) != 0:
                            self.agent.plot_logs()
                            log_manager.update(players)
                            log_manager.plot_logs()
            self.agent.save_weights()
            log_manager.update(players)
            log_manager.plot_logs()
            log_manager.plot_best_frames()

if __name__ == '__main__':
    pass
