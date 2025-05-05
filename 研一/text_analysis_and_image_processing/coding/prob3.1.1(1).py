import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转为灰度图
img = cv2.imread("./coding/img.jpg", cv2.IMREAD_GRAYSCALE)

def show(title: str, image):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"./coding/3.1.1_results/{title.lower().replace(' ', '_')}.png", dpi=100)
    plt.show()

show("Origin", img)

# 领域平均滤波（均值滤波）
mean_blur = cv2.blur(img, (5, 5))
show("Mean Blur", mean_blur)

# Gauss滤波
gaussian_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1)
show("Gaussian Blur", gaussian_blur)

# 一阶微分算子（Sobel边缘检测）
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
show("Sobel Gradient Magnitude", sobel)

# 二阶微分算子（Laplacian）
laplacian = cv2.Laplacian(img, cv2.CV_64F)
show("Laplacian", laplacian)

# 非锐化掩模（Unsharp Mask）
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
unsharp_mask = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
show("Unsharp Mask", unsharp_mask)
