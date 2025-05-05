import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, io
config = {
    "font.family": 'serif', # 衬线字体
    "figure.figsize": (14, 6),  # 图像大小
    "font.size": 20, # 字号大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'cm', # 渲染数学公式字体
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

# 读取图像并添加噪声
img = cv2.imread("./coding/img.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# noisy_img = img.astype(np.float32) / 255.0 + np.random.normal(0, 0.05, img.shape)
noisy_img = img.astype(np.float32) / 255.0 + np.random.normal(0, 0.2, img.shape)
noisy_img = np.clip(noisy_img, 0, 1)

# 1. 双边滤波（OpenCV）
bilateral_filtered = cv2.bilateralFilter((noisy_img * 255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)

# 2. 非局部均值滤波（scikit-image）
img_float = img_as_float(noisy_img)
sigma_est = np.mean(estimate_sigma(img_float, channel_axis=-1))
nlm_filtered = denoise_nl_means(img_float, h=1.15 * sigma_est, fast_mode=True,
                                 patch_size=5, patch_distance=6, channel_axis=-1)

# 显示结果
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(img)
axs[0].set_title("原图")
axs[1].imshow(noisy_img)
axs[1].set_title("含噪图")
axs[2].imshow(bilateral_filtered)
axs[2].set_title("双边滤波")
axs[3].imshow(nlm_filtered)
axs[3].set_title("非局部均值滤波")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.savefig("./coding/5.2.1_results/output.png", dpi=100)
plt.show()
