import numpy as np
def perceptron(xn, yn, eta=0.7, max_iter=2000, w=np.zeros(3)):
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
    return w
