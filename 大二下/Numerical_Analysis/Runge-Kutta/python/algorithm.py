import numpy as np

def Runge_Kutta(f, x, y, h, N):
    """
    Args:
        f: n+1维函数, 输入方式为f(x, y), nx1
        x: 初始值x0, 1x1
        y: 初始值y0, nx1
        h: 步长
        N: 计算步数
    Returns: Runge_Kutta法计算N步后的值y(xN), nx1
    """
    for _ in range(N):
        K1 = h * f(x, y)
        K2 = h * f(x + h/2, y + K1/2)
        K3 = h * f(x + h/2, y + K2/2)
        K4 = h * f(x + h, y + K3)
        y = y + (K1 + 2*K2 + 2*K3 + K4) / 6
        x = x + h
        print(K1, K2, K3, K4)
        print(y)
    return y
