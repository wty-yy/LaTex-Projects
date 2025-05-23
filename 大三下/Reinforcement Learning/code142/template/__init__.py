# -*- coding: utf-8 -*-
'''
@File    : __init__.py
@Time    : 2023/04/04 10:13:11
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 模板初始化文件
'''

if __name__ == '__main__':
    pass

import matplotlib.pyplot as plt

def set_plot_config():
    config = {  # matplotlib绘图配置
        "figure.figsize": (6, 6),  # 图像大小
        "font.size": 16, # 字号大小
        "font.sans-serif": ['SimHei'],   # 用黑体显示中文
        'axes.unicode_minus': False, # 显示负号
        "mathtext.fontset": 'stix', # 渲染数学公式字体
    }
    plt.rcParams.update(config)

set_plot_config()

if __name__ == "__main__":
    print('start init')