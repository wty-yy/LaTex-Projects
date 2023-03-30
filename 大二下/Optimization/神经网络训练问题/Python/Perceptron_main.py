# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

config = {  # 图像配置
    "figure.figsize": (8, 8),  # 固定图像大小
    "figure.dpi": 200,
    "font.family": 'serif',
    "font.size": 24,
    "savefig.bbox": 'tight',  # 减小边框大小
    "mathtext.fontset": "stix",
    "font.serif": ['SimSun'],
    "axes.unicode_minus": False,  # 用来正常显示负号
}
plt.rcParams.update(config)

# 生成100个随机点
N = 100
xn = np.random.rand(N, 2)  # 第一列为横轴,第二列为纵轴
x = np.linspace(0, 1)  # 选取[0,1]上的线性分布

# 选取初始分割直线
a, b = -0.6, 0.8
f = lambda x: a * x + b
yn = np.zeros([N, 1])  # 样本的标签集,点在分割线上方为1,反之为-1

def print_example(xn, yn):  # 打印数据集
    plt.plot(xn[:, 0], xn[:, 1], 'o', color='tab:blue')
    for i in range(N):
        if f(xn[i, 0]) > xn[i, 1]:  # 分割线下方
            yn[i] = 1
            plt.plot(xn[i, 0], xn[i, 1], 'o', color='tab:green')
        else:
            yn[i] = -1  # 分割线上方

plt.plot(x, f(x), 'tab:red')
print_example(xn, yn)
plt.legend(['分离超平面', '正类', '负类'])
plt.savefig('分离平面.pdf')
plt.show()

def print_hyper(xn, yn, w):  # 打印超平面
    if w[2] == 0:
        return
    y = lambda x: -w[0] / w[2] - w[1] / w[2] * x
    x = np.linspace(0, 1)
    plt.plot(x, y(x), 'tab:blue')
    print_example(xn, yn)

def perceptron(xn, yn, eta=0.7, max_iter=2000, w=np.random.rand(3)):
    """
    Input
        xn: 样本的特征, Nx2矩阵
        yn: 样本的标签, Nx1矩阵
        eta: 学习率
        max_iter: 最大迭代次数
        w: 初始化参数
    Output
        w: 迭代结果,最优分类曲线
    """
    f = lambda x: np.sign(w[0] + w[1] * x[0] + w[2] * x[1])  # 当前点x在直线的上方则返回1
    for _ in range(max_iter):
        i = np.random.randint(N)  # 随机选取一个样本
        if yn[i] != f(xn[i, :]):  # 如果该样本为误分类点,则进行修正曲线
            w[0] += eta * yn[i]
            w[1] += eta * yn[i] * xn[i, 0]
            w[2] += eta * yn[i] * xn[i, 1]
        if _ + 1 == 1 or _ + 1 == 100 or _ + 1 == 500 or _ + 1 == 2000:
            print_hyper(xn, yn, w)
            plt.savefig('迭代' + str(_+1) + '.pdf')
            plt.show()
    return w

w = perceptron(xn, yn)  # 开始感知机学习算法

print_example(xn, yn)
y = lambda x: -w[0] / w[2] - w[1] / w[2] * x
plt.plot(x, y(x), 'b--', label='分离超平面')
plt.plot(x, f(x), 'r', label='初始分割线')
plt.legend()
plt.savefig('对比图.pdf')
plt.show()
