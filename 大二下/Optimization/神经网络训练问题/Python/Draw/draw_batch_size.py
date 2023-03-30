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

# 网络结构: 784-16-16-10, 使用Sigmoid作为激活函数, dropout=0, Batch Size=30, 50, 80, 100, 200, 500, 训练次数: 500
with open('diff_batch_size.txt', 'r', encoding='utf-8') as file:
    file.readline()
    for _ in range(6):
        y = file.readline().split(' ')
        y = [float(k) for k in y]
        x = range(len(y))
        plt.plot(x, y)
    plt.axis([-10, 500, 0.05, 0.95])
    plt.title('784-16-16-10, Sigmoid')
    plt.xlabel('训练次数/epoch')
    plt.ylabel('准确率')
    plt.legend(['Batch size=30', 'Batch size=50', 'Batch size=80', 'Batch size=100', 'Batch size=200', 'Batch size=500'])
    plt.savefig('diff_batch_size.pdf')
    plt.show()
