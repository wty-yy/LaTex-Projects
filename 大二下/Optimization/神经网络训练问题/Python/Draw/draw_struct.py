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

# 网络结构: 784-16, 32, 64, 128-10, 使用tanh, ReLU作为激活函数, dropout=0, Batch Size=100, 训练次数: 500
with open('diff_struct.txt', 'r', encoding='utf-8') as file:
    file.readline()
    for _ in range(8):
        y = file.readline().split(' ')
        y = [float(k) for k in y]
        x = range(len(y))
        plt.plot(x, y)
    plt.axis([-10, 500, 0.35, 0.95])
    plt.title('tanh, ReLU, Batch Size=100')
    plt.xlabel('训练次数/epoch')
    plt.ylabel('准确率')
    plt.legend(['784-16-10 Tanh', '784-32-10 Tanh', '784-64-10 Tanh', '784-128-10 Tanh',
                '784-16-10 ReLU', '784-32-10 ReLU', '784-64-10 ReLU', '784-128-10 ReLU'], ncol=2)
    plt.savefig('diff_struct.pdf')
    plt.show()
