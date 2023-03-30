import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """
    用于读取MNIST数据集, 代码参考: https://blog.csdn.net/simple_the_best/article/details/75267863
    Args:
        path: 读入文件的路径
        kind: 读入数据类型, train为训练数据, t10k为测试数据
    Returns: 像素矩阵 nx784 和标签向量 nx1
    """
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:  # 'rb'以二进制形式读入标签文件
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)  # 转化为图像矩阵28*28
    return images, labels

def print_same(X, y, num=0, line=2):
    """
    Args:
        X: 图像矩阵
        y: 标签向量
        num: 打印的数字
        line: 打印行数
    Returns: 打印图像矩阵中标签为num的前line*5个图片
    """
    fig, ax = plt.subplots(nrows=line, ncols=5, sharex=True, sharey=True)

    ax = ax.flatten()
    for i in range(line * 5):
        # 将序号为7的图像切片取出, 每一行都是一个7图片的数据, 然后取前i个即可
        img = X[y == num][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X_train, y_train = load_mnist(os.getcwd())
    print_same(X_train, y_train, num=6, line=3)
