# -*- coding: utf-8 -*-
'''
@File    : sga.py
@Time    : 2023/04/28 11:29:27
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 简单遗传算法SGA在[-1,2]上最大化f(x)=x*sin(4*pi*x)+x**2，精确到6位小数
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution  # 比较结果误差
from pathlib import Path
PATH_FIGURES = Path(__file__).parent  # 当前代码文件夹作为图片路径

def show_scipy_min():
    g = lambda x: -1 * f(x)
    res = differential_evolution(g, bounds=[(-1, 2)], tol=1e-6)
    print(f"scipy: 最优值 f({res.x[0]:.6f}) = {-res.fun:.6f}")
    return -res.fun

class SGA:
    def __init__(self, xmin, xmax, func, delta=1e-6, M=30, pc=0.8, pm=0.005, N=200):
        self.xmin, self.xmax, self.func, self.delta = xmin, xmax, func, delta
        self.M, self.pc, self.pm, self.N = M, pc, pm, N  # 超参数
        self.best = {'f': -np.inf, 'x': 0}
        self.logs = {'p_means': [], 'f_means': []}
        # 1. 计算编码长度L
        self.L = np.ceil(np.log2((self.xmax-self.xmin) / self.delta + 1)).astype(int)

    def decode(self, y):
        return self.xmin + (self.xmax - self.xmin) / ((1<<self.L) - 1) * y
    
    def solve(self):
        # 2. 随机产生初始群体
        y = np.random.randint(0, 1<<self.L, size=self.M)
        for _ in range(self.N):
            # 3. 计算适应度值（得分）
            s = self.func(self.decode(y))
            if np.max(s) > self.best['f']:
                self.best['f'] = np.max(s)
                self.best['x'] = self.decode(y[np.argmax(s)])
            self.logs['p_means'].append(np.mean(y))
            self.logs['f_means'].append(np.mean(s))
            # 4. 每个个体的选择概率（概率分布）加上1e-8是避免除0
            p = (s - s.min() + 1e-8) / np.sum(s - s.min() + 1e-8)
            y = y[np.random.choice(self.M, self.M, p=p)]
            # 5. 交叉操作
            idxs = np.random.choice(self.M, int(self.M * self.pc // 2 * 2))
            for i in range(int(self.M * self.pc // 2)):
                a, b = y[idxs[i<<1]], y[idxs[i<<1|1]]
                qc = np.random.randint(0, self.L) + 1  # 交换前qc位
                c, d = a.copy(), b.copy()
                a = ((a >> qc) << qc) + (d & ((1<<qc) - 1))
                b = ((b >> qc) << qc) + (c & ((1<<qc) - 1))
                y[idxs[i<<1]], y[idxs[i<<1|1]] = a, b
            # 6. 变异操作
            for i in range(len(y)):
                for j in range(self.L):
                    if np.random.rand() <= self.pm:
                        y[i] ^= (1<<j)
        print(f"SGA: 最优值 f({self.best['x']:.6f}) = {self.best['f']:.6f}")
        return self.best['f']

def f(x): return x * np.sin(4 * np.pi * x) + x ** 2

if __name__ == '__main__':
    sga = SGA(-1, 2, func=f)
    sga_best = sga.solve()
    true_best = show_scipy_min()
    print(f"误差: {np.abs(sga_best - true_best)}")
    
    logs = sga.logs
    fig, ax = plt.subplots(figsize=(8, 5))
    means = np.array(logs['p_means'])
    means = (means - means.min()) / (means.max() - means.min())
    ax.plot(logs['f_means'], '-*', label='f mean')
    ax.plot(means + 4.5, '-*', label='Population mean')
    ax.plot([0, 500], [4.300863, 4.300863], '--r', label='True best')
    ax.set_xlim(0, 25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATH_FIGURES.joinpath("plot_sga.png"), dpi=300)
    plt.show()