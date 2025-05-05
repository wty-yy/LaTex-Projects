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

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = image.copy()
    h, w = noisy.shape
    num_salt = int(salt_prob * h * w)
    num_pepper = int(pepper_prob * h * w)

    # 添加盐噪声（白点）
    coords = [np.random.randint(0, i, num_salt) for i in noisy.shape]
    noisy[coords[0], coords[1]] = 255

    # 添加椒噪声（黑点）
    coords = [np.random.randint(0, i, num_pepper) for i in noisy.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

# 创建椒盐噪声
img_noisy = add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)
show("Image with Salt-and-Pepper Noise", img_noisy)

# 中值滤波
median_denoised = cv2.medianBlur(img_noisy, 3)
show("Denoised with Median Filter", median_denoised)

# 最大值滤波
kernel = np.ones((3,3), np.uint8)
max_filter = cv2.dilate(median_denoised, kernel)
show("Denoised with Max Filter", max_filter)

# 最小值滤波
min_filter = cv2.erode(median_denoised, kernel)
show("Denoised with Min Filter", min_filter)
