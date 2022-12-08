import numpy as np
x = []
import json

def prob26():
    with open(r".\tmp.txt") as file:
        for line in file.readlines():
            x.append(json.loads(line))
    mean = np.mean(x, axis=-1)
    std = np.std(x, ddof=1, axis=-1)
    var = np.power(std, 2)
    sw = (5 * var[0] + 5 * var[1]) / 10
    print((mean[1] - mean[0]) / (np.sqrt(sw * 1/3)))

def prob14():
    x = np.array([[20.5, 18.8, 19.8, 20.9, 21.5, 19.5, 21, 21.2], [17.7, 20.3, 20, 18.8, 19, 20.1, 20, 19.1]])
    mean = np.mean(x, axis=-1)
    std = np.std(x, axis=-1, ddof=1)
    var = np.power(std, 2)
    sw = 7 * (var[0] + var[1]) / 14
    print((mean[1] - mean[0]) / (sw / 2))

def prob27():
    n = np.array([22, 24])
    mean = np.array([2.36, 2.55])
    std = np.array([0.57, 0.48])
    var = np.power(std, 2)
    sw = np.sum((n-1) * var) / 44
    print((mean[1]-mean[0]) / (sw * np.sqrt(1/n[0]+1/n[1])))

if __name__ == '__main__':
    prob27()