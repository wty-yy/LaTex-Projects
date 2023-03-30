import numpy as np
import algorithm

def f(x, y):  # 向量函数f
    return np.array([y[1], y[2], y[2] + y[1] - y[0] + 2*x - 3])

def g(x):  # 解析解
    return x * np.exp(x) + 2*x - 1

x = 0
y = np.array([-1, 3, 2])
h = 0.05
N = 20
result = algorithm.Runge_Kutta(f, x, y, h, N)[0]
ans = g(1)
print('y(1)处的解分别为')
print('Runge-Kutta:', result)
print('解析解:', ans)
print('误差:', np.abs(ans - result))
