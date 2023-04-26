# -*- coding: utf-8 -*-
'''
@File    : main.py
@Time    : 2023/04/20 17:55:25
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 启动程序
'''

if __name__ == '__main__':
    pass

from swapper import Swapper
import player
import agent

def main():
    swapper = Swapper(episodes=100)
    # swapper.agent.load_weights(agent.PATH_CHECKPOINT.joinpath("complete/dqn_best0"))
    # swapper.start(is_train=True)
    swapper.start(is_train=True, is_render=True)
    # swapper.start(is_train=False)

def test():
    swapper = Swapper(episodes=100)
    swapper.agent.load_weights(agent.PATH_CHECKPOINT.joinpath("complete/dqn_best1"))
    swapper.start(is_train=False, is_render=True)

def continues():
    swapper = Swapper(episodes=100)
    swapper.agent.load_weights(agent.PATH_CHECKPOINT.joinpath("n_player=6,lr=0.001,batch=32/dqn_best1"))
    # swapper.start(is_train=True)
    swapper.start(is_train=True, is_render=True)

if __name__ == '__main__':
    main()
    # continues()
    # test()
