# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sga import SGA
from scipy.optimize import differential_evolution  # 比较结果误差
from plot_figure import plot_sga, PATH_FIGURES
    
def f(x): return x * np.sin(4 * np.pi * x) + x ** 2

def single_test():
    sga = SGA(-1, 2, func=f)
    sga_best = sga.solve()
    true_best = show_scipy_min()
    print(f"误差: {np.abs(sga_best - true_best)}")
    plot_sga(sga.logs)

def show_scipy_min():
    g = lambda x: -1 * f(x)
    res = differential_evolution(g, bounds=[(-1, 2)], tol=1e-6)
    print(f"scipy: 最优值 f({res.x[0]:.6f}) = {-res.fun:.6f}")
    return -res.fun

def grid_test():
    total = 50  # 取total次执行sga的均值
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    grid = {
        'M': [30, 50, 100],
        'pc': [0.1, 0.5, 1],
        'pm': [0.005, 0.01]
    }
    for idx, key in enumerate(grid.keys()):
        ax = axs[idx]
        for arg in tqdm(grid[key]):
            kwarg = {key: arg, 'verbose': False}
            sga = SGA(-1, 2, func=f, **kwarg)
            logs, name = np.zeros(sga.N), f"{key}={arg}"
            for _ in range(total):
                sga.reset(); sga.solve()
                logs += np.array(sga.logs['f_means'])
            ax.plot(logs / total, '-*', label=f'{name}')
        ax.plot([0, 500], [4.300863, 4.300863], '--r', label='True best')
        ax.set_xlim(0, 50)
        ax.legend()
    fig.suptitle("Mean of f(x) (default: M=30,pc=0.8,pm=0.005)")
    fig.tight_layout()
    fig.savefig(PATH_FIGURES.joinpath("parameter_compare.png"), dpi=300)
    plt.show()

if __name__ == '__main__':
    single_test()
    # grid_test()

