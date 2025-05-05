import os
from PIL import Image
import matplotlib.pyplot as plt

# 输入图像路径
input_path = "./coding/img.jpg"
basename = os.path.splitext(os.path.basename(input_path))[0]

# 输出目录路径
output_dir = "./coding/1.2.1_results"
os.makedirs(output_dir, exist_ok=True)

# 读取图像
img = Image.open(input_path)

# 显示图像
plt.imshow(img)
plt.title(f"Origin Image: {input_path}")
plt.axis('off')
plt.tight_layout()
plt.show()

# 保存为不同格式
output_formats = ["PNG", "JPEG", "BMP", "TIFF", "WEBP", "PDF"]
output_paths = []

for fmt in output_formats:
    output_file = os.path.join(output_dir, f"{basename}.{fmt.lower()}")
    img.save(output_file, fmt)
    output_paths.append(output_file)

# 比较文件大小
print(f"{'format':<10} {'size(KB)':>10}")
print("-" * 22)

# 原图像大小
original_size = os.path.getsize(input_path) / 1024
print(f"{'JPG(ORG)':<10} {original_size:10.2f}")

for path in output_paths:
    fmt = os.path.splitext(path)[1][1:].upper()
    size_kb = os.path.getsize(path) / 1024
    print(f"{fmt:<10} {size_kb:10.2f}")
