import matplotlib.pyplot as plt
config = {  # matplotlib绘图配置
  "figure.figsize": (6, 6),  # 图像大小
  "font.size": 16, # 字号大小
  "font.sans-serif": ['SimSun'],   # 用黑体显示中文
  'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)
import pandas as pd
import numpy as np

draw_config = {
  'cols': [
    'Trifinger__sac__7__2024-05-15 21:39:17 - reward',
    'trifinger__td3__7__2024-05-13 20:17:15 - reward',
    'trifinger__ddpg__7__2024-05-13 20:13:12 - reward',
    'Ant__ppo_tt__7__1713082913 - reward'
  ],
  'labels': [
    'SAC', 'TD3', 'DDPG', 'PPO'
  ],
  'x_cvt': [
    lambda x: x * 64.53 / np.max(x),
    # lambda x: x * 64.063 / np.max(x),
    lambda x: x * 28.229 / np.max(x),
    lambda x: x * 64.136 / np.max(x),
    lambda x: x * 28.229 / np.max(x),
  ]
}
class Drawer:
  def __init__(self, path_csv):
    self.df = pd.read_csv(path_csv)
    print(self.df.columns)
  
  def draw(self, cfg: dict = draw_config):
    xclip = np.array([-np.inf, np.inf], np.float32)
    fig, ax = plt.subplots(figsize=(8,5))
    for col, label, fn in zip(*cfg.values()):
      y = self.df[col]
      if label == 'TD3': y *= 10
      n = (~np.isnan(y)).sum()
      y = y[:n]
      x = fn(self.df['Step'][:n])
      xclip[0] = max(xclip[0], x.min())
      xclip[1] = min(xclip[1], x.max())
      ax.plot(x, y, label=label)
    ax.set_xlim(xclip)
    ax.set_xlabel("时间（单位/小时）")
    ax.set_xticks(np.arange(int(xclip[1]+1))[::4])
    ax.set_ylabel("平均奖励")
    plt.tight_layout()
    plt.legend()
    plt.savefig("Image.png", dpi=300)
    plt.show()

if __name__ == '__main__':
  drawer = Drawer("wq.csv")
  drawer.draw()

