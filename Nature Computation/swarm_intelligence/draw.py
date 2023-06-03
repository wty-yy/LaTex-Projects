import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['font.family'] = ['serif']
PATH_DATA = Path(__file__).parent

with open(PATH_DATA.joinpath("citys.txt")) as file:
    n = int(file.readline()); X = np.loadtxt(file)

plt.subplots(figsize=(5, 5))
with open(PATH_DATA.joinpath("best_path.txt")) as file:
    path = np.loadtxt(file, dtype=np.int32)
path_pos = np.array([X[i] for i in path])
dis = 0
for i in range(path_pos.shape[0]-1):
    dis += np.sqrt(np.sum(np.power(path_pos[i]-path_pos[i+1], 2)))
plt.plot(path_pos[:,0], path_pos[:,1])
plt.plot(X[:,0], X[:,1], ".", markersize=10)
plt.title(f"Distance: {dis:.4f}")
plt.tight_layout()
plt.savefig(PATH_DATA.joinpath("ant_best_path.png"), dpi=300)
plt.show()