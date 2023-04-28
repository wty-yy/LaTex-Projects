# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
from pathlib import Path
PATH_FIGURES = Path(__file__).parent

def f(x): return x * np.sin(4 * np.pi * x) + x ** 2

def plot_f(xmin=-1, xmax=2):
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.linspace(xmin, xmax, 10000)
    ax.plot(x, f(x))
    x_ = x[np.argmax(f(x))]
    ax.plot(x_, f(x_), 'r.', markersize=10, label="maximum")
    ax.plot([-1, 2], [0, 0], 'k--')
    ax.plot([0, 0], [-2, 10], 'k--')
    ax.set_xlim(-1, 2)
    ax.set_ylim(-0.3, 4.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATH_FIGURES.joinpath("plot_f.png"), dpi=300)
    plt.show()

def plot_sga(logs):  # 'means', 'best'
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

def main():
    plot_f()
    
if __name__ == '__main__':
    main()