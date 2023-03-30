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

# 网络结构: 784-32-10, 使用Sigmoid作为激活函数, Batch Size=100, dropout=0, 总训练次数: 500
with open('diff_early_stop.txt', 'r', encoding='utf-8') as file:
    file.readline()
    for _ in range(1):
        y = file.readline().split(' ')
        y = [float(k) for k in y]
        x = range(len(y))
        plt.plot(x, y)
    z = np.polyfit(x, y, 7)
    p = np.poly1d(z)
    yy = p(x)
    idx = yy.argmax()
    print('拟合曲线最大值点: ', x[idx])
    plt.plot(x, yy)
    plt.plot(x[idx], yy[idx], 'o', color='r')
    plt.axis([-10, 500, 0.75, 0.95])
    plt.title('784-32-10, Sigmoid, Batch Size=100')
    plt.xlabel('训练次数/epoch')
    plt.ylabel('准确率')
    plt.legend(['真实结果', '拟合曲线', '最大值点'])
    plt.savefig('diff_early_stop.pdf')
    plt.show()
