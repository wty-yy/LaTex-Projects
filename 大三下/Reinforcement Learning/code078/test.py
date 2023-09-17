import numpy as np
import matplotlib.pyplot as plt

config = {  # matplotlib绘图配置
    "figure.figsize": (6, 6),  # 图像大小
    "font.size": 16, # 字号大小
    "font.sans-serif": ['SimHei'],   # 用黑体显示中文
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

# 可视化状态价值
T = 4
value = np.loadtxt("value4.txt", dtype=np.int32)
ax = plt.figure(figsize=(5, 5)).add_subplot(projection='3d')
x1, x2 = np.meshgrid(  # 创建离散网格点
    np.linspace(0, 20, 1000),
    np.linspace(0, 20, 1000)
)
z = np.zeros_like(x1)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        z[i, j] = value[round(x2[i, j]), round(x1[i, j])]
ax.plot_surface(x1, x2, z)
ax.set_xlabel("B地汽车数目")
ax.set_ylabel("A地汽车数目")
plt.tight_layout()
plt.savefig(f"value{T}.png", dpi=300)
plt.show()