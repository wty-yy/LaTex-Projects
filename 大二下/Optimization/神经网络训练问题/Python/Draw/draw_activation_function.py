import numpy as np
import matplotlib.pyplot as plt

config = {  # 图像配置
    "figure.figsize": (15, 8),  # 固定图像大小
    "figure.dpi": 200,
    "font.family": 'serif',
    "font.size": 24,
    "savefig.bbox": 'tight',  # 减小边框大小
    "mathtext.fontset": "stix",
    "font.serif": ['SimSun'],
    "axes.unicode_minus": False,  # 用来正常显示负号
}
plt.rcParams.update(config)

# 网络结构: 784-32-10, 使用Sigmoid, Tanh, ReLU作为激活函数, dropout=0, Batch Size=100, 训练次数: 500
with open('diff_activation_function.txt', 'r', encoding='utf-8') as file:
    file.readline()
    for _ in range(3):
        y = file.readline().split(' ')
        y = [float(k) for k in y]
        x = range(len(y))
        plt.plot(x, y)
    plt.axis([-10, 500, 0.55, 0.95])
    plt.title('784-32-10, Batch Size=100')
    plt.xlabel('训练次数/epoch')
    plt.ylabel('准确率')
    plt.legend(['Sigmoid', 'Tanh', 'ReLU'])
    plt.savefig('diff_activation_function.pdf')
    plt.show()
