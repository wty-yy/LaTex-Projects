from template import reinforcement
import matplotlib.pyplot as plt
import numpy as np

def plot_test():
    np.random.seed(42)
    m = int(1e5)
    X_normal = np.random.randn(m, 1)
    hists, bins = np.histogram(X_normal, bins=50, density=True)
    bins = bins[:-1] + (bins[1] - bins[0]) / 2  # 取每个小区间的中位数

    plt.figure(figsize=(8, 4))
    plt.hist(X_normal, bins=50, density=True, alpha=0.8)  # 绘制直方图，alpha为透明度
    plt.plot(bins, hists, 'r--o', markersize=6, label="高斯采样")
    plt.text(2.5, 0.3, r"$f(x) = \frac{exp\left(\frac{-x^2}{2}\right)}{\sqrt{2\pi}}$",
            fontsize=25, ha='center', color='k', math_fontfamily='cm')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.axis([-4, 4, -0.02, 0.42])
    plt.xlabel("$x$")
    plt.ylabel("$P$", rotation=0)
    plt.legend(loc='upper left')
    plt.title("$10^5$个来自标准正态分布的样本")
    plt.tight_layout()
    plt.show()
    print("曲线下近似面积:", np.trapz(hists, bins))  # 0.9999800000000001

if __name__ == '__main__':
    from pathlib import Path
    print(Path(__file__).parent)