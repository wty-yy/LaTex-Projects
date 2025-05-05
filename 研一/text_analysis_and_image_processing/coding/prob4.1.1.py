import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像并转为灰度图
img_path = "./coding/img.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)

# 2. 执行FFT变换并移动低频到中心
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 3. 构造滤波器（低通和高通）
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
radius = 30  # 半径

# 低通滤波器
low_pass_mask = np.zeros_like(img)
cv2.circle(low_pass_mask, (ccol, crow), radius, 1, thickness=-1)

# 高通滤波器 = 1 - 低通滤波器
high_pass_mask = 1 - low_pass_mask

# 应用滤波器
f_low = fshift * low_pass_mask
f_high = fshift * high_pass_mask

# 逆FFT变换
img_low = np.fft.ifft2(np.fft.ifftshift(f_low))
img_low = np.abs(img_low)

img_high = np.fft.ifft2(np.fft.ifftshift(f_high))
img_high = np.abs(img_high)

# 4. 可视化结果
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(magnitude_spectrum, cmap='gray')
axs[0, 1].set_title('FFT Magnitude Spectrum')
axs[0, 1].axis('off')

axs[0, 2].imshow(low_pass_mask*magnitude_spectrum, cmap='gray')
axs[0, 2].set_title('Low-pass Mask')
axs[0, 2].axis('off')

axs[1, 0].imshow(img_low, cmap='gray')
axs[1, 0].set_title('Low-pass Filtered Image')
axs[1, 0].axis('off')

axs[1, 1].imshow(high_pass_mask*magnitude_spectrum, cmap='gray')
axs[1, 1].set_title('High-pass Mask')
axs[1, 1].axis('off')

axs[1, 2].imshow(img_high, cmap='gray')
axs[1, 2].set_title('High-pass Filtered Image')
axs[1, 2].axis('off')

plt.tight_layout()
plt.savefig("./coding/4.1.1_results/fft_results.png", dpi=100)
plt.show()
