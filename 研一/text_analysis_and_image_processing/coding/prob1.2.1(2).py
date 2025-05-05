import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
input_path = "./coding/img.jpg"
img = Image.open(input_path).convert("RGB")
img_np = np.array(img)

# Print image info
H, W, C = img_np.shape
print("Image shape (Height x Width x Channels):", img_np.shape)

# Calculate center and extract 40x40 patch
center_y, center_x = H // 2, W // 2
half = 20
img_block = img_np[center_y - half:center_y + half, center_x - half:center_x + half]
# print("40x40 pixel matrix at the center of the image:")
# print(img_block)

# Show the 40x40 center block
plt.imshow(img_block)
plt.title("40x40 Center Block")
plt.axis('off')
plt.tight_layout()
plt.savefig("./coding/1.2.1_results/40x40_center_block.png", dpi=100)
plt.show()

# Show separate RGB channels
channels = ['Red Channel', 'Green Channel', 'Blue Channel']
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

for i in range(3):
    axs[i].imshow(img_np[:, :, i], cmap='gray')
    axs[i].set_title(channels[i])
    axs[i].axis('off')

plt.tight_layout()
plt.savefig("./coding/1.2.1_results/show_diff_channels.png", dpi=100)
plt.show()

# Print partial red channel matrix values
print("Top-left 5x5 pixel values of the red channel:")
print(img_np[:5, :5, 0])
