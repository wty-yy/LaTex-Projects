import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif']
PATH_FIGURE = Path(__file__).parent

np.random.seed(42)
rand = np.random.rand; randint = np.random.randint  # 重载随机函数名
citys = np.loadtxt(Path(__file__).parent.joinpath("data.txt"))  # 数据读入
n = citys.shape[0]
def getdis(path):
    dis = 0;
    for i in range(n):
        dis += np.sqrt(np.sum(np.power(citys[path[i]] - citys[path[(i+1)%n]], 2)))
    return dis

def plot(path, cnt, ax:plt.Axes):
    for i in range(n):
        x, y = citys[path[i],:], citys[path[(i+1)%n],:]
        ax.plot([x[0],y[0]], [x[1],y[1]], '-b')
    ax.plot(citys[:,0], citys[:,1], '.r', markersize=10)
    ax.text(0.15, 0.83, f"times: {cnt}\ndistence:{getdis(path):.4}", fontsize=15)
    ax.set_xticks([]); ax.set_yticks([])

fig, axs = plt.subplots(2, 2, figsize=(8, 6)); ax = iter(axs.reshape(-1))
# plot(list(range(10)), 0, axs[0])
# fig.tight_layout(); plt.show()
# exit()
T = 168.14; K = 1; delta = 0.999; cnt = 0
path = list(range(n)); now = getdis(path)
best = {'path': path.copy(), 'dis': now}
while T >= 0.01:
    r = rand(); path_ = path.copy()
    if r <= 1/3:
        a, b = randint(0, n, size=2)  # 交换a和b
        while b == a: b = randint(0, n)
        path[a], path[b] = path[b], path[a]
    elif r <= 2/3:  # 使[s,s+l)区间段逆序
        s, l = randint(0,n,size=2); l = min(l, n-s); path[s:s+l] = reversed(path[s:s+l])
    else:  # 将path[p]插入到位置q的前面
        p, q = randint(0,n,size=2); idx = path[p]
        path = path[0:p] + path[p+1:]
        path = path[0:q] + [idx] + path[q:]
    new = getdis(path)
    if np.exp((now - new)/(T*K)) > rand(): now = new
    else: path = path_
    if now < best['dis']: best['dis'] = now; best['path'] = path.copy()
    T *= delta; cnt += 1
    if cnt in [1, 100, 5000]: plot(best['path'], cnt, next(ax))
plot(best['path'], cnt, next(ax))
print(f"最优路径: {best['path']}, 长度: {best['dis']}")
fig.tight_layout()
# fig.savefig(PATH_FIGURE.joinpath("TSP.png"), dpi=300); plt.show()
plt.show()
