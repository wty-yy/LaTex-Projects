# coding=utf-8
import numpy as np
from tabulate import tabulate

def Romberg(fun, a, b, eps):
    """
    Romberg积分法
    :param fun: 一维被积函数
    :param a: 积分区间左端点
    :param b: 积分区间右端点
    :param eps: 截断误差
    :return: 积分结果
    """
    upper = 20  # 计算次数的上界
    table = [[(b-a)/2 * (fun(a) + fun(b))]]  # T1初始化
    for i in range(1, upper):
        tmp = 0
        for k in range(np.power(2, i-1)):
            tmp += fun(a + (2*k+1) * (b-a) / np.power(2, i))
        tmp *= (b - a) / np.power(2, i)
        table.append([1/2 * table[i-1][0] + tmp])  # 计算T_2^i
        for j in range(1, min(i, 3) + 1):  # 递归求值
            table[i].append(table[i][j-1] + (table[i][j-1]-table[i-1][j-1]) / (np.power(4, j)-1))
        if i >= 4 and np.abs(table[i][3] - table[i - 1][3]) <= eps:  # 达到精度要求
            break
    print(tabulate(table, tablefmt='latex', floatfmt=".12f"))  # 以LaTex格式输出表格
    return table[i][3]


if __name__ == '__main__':
    print(Romberg(lambda x: 1/(1+x), 0, 1, 1e-15))
    # print(Romberg(lambda x: np.log(1+x)/(1+np.power(x,2)), 0, 1, 1e-15))
    # print(Romberg(lambda x: np.log(1+x)/x, 1e-30, 1, 1e-10))
    # print(Romberg(lambda x: np.sin(x)/x, 1e-30, np.pi/2, 1e-15))