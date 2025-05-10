import numpy as np

def prob2():
    p = np.array([[1/2,1/3,1/6], [0,1/3,2/3], [1/2,0,1/2]])
    p = p @ p @ p
    print(p)
    pi = np.array([1/4, 1/4, 1/2]).reshape(1,3)
    print(pi.shape)
    print(pi @ p @ np.array([0, 1, 2]))
    print(53/54)

def prob6():
    base = np.array([[0.4,0.6],[0.5,0.5]])
    p = base
    for _ in range(4):
        p @= base
    print(p[0,1])
    print(6/11)

if __name__ == '__main__':
    # prob2()
    prob6()