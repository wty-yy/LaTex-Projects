import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.io import imread

# Load image (grayscale)
img_path = "./coding/low_light.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Histogram Equalization
equalized = cv2.equalizeHist(img)

# Histogram Specification (match to bright reference)
# For demonstration, use a synthetic reference histogram
reference = np.linspace(0, 255, img.shape[0] * img.shape[1]).reshape(img.shape).astype(np.uint8)
specified = exposure.match_histograms(img, reference, channel_axis=None)

# Plot results
titles = ["Original Image", "Histogram Equalization", "Histogram Specification"]
images = [img, equalized, specified]

plt.figure(figsize=(12, 3))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("./coding/2.2.1_results/processed_results.png", dpi=100)
plt.show()

# Plot histograms
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.hist(images[i].ravel(), bins=256, range=[0, 256], color='gray')
    plt.title(titles[i] + " Histogram")
plt.tight_layout()
plt.savefig("./coding/2.2.1_results/histogram_results.png", dpi=100)
plt.show()
