import os
import numpy as np
import Input  # 数据读入
import ANN  # 神经网络核心部分


def change_output(y_output):
    #  转化输出为列向量
    output = []
    for i in y_output:
        tmp = [0] * 10
        tmp[i] = 1
        output.append(tmp)
    return np.array(output)

if __name__ == '__main__':
    X_train, y_train = Input.load_mnist(os.getcwd())
    X_check, y_check = Input.load_mnist(os.getcwd(), 't10k')
    X_train = X_train / 255
    X_check = X_check / 255
    y_train = change_output(y_train)
    y_check = change_output(y_check)

    dropout = 0
    #  创建神经网络模型
    model = ANN.Model(
        ANN.Layer(784, 32, activation='tanh', dropout=dropout),  # 神经网络结构
        ANN.Layer(32, 10),
        loss='sse',
        test_input=X_check,  # 测试集用于计算准确率, 不用于训练
        test_output=y_check,
        test_rate=0.1,  # 如果存储每次梯度更新后的准确度, 即memory=1, 计算准确度所需的测试集大小占比
    )
    batch = 100  # batch_size
    model.fit(X_train, y_train, epoch=10, batch=batch, memory=True)
    end_accuracy = model.evaluate(test_rate=1)
    print(end_accuracy)  # 输出最终准确率

    if model.memory:  # 保存准确率
        with open('diff_accuracy.txt', 'a', encoding='utf-8', errors='ignore') as file:
            file.write('网络结构: 784-32-10, 使用tanh作为激活函数, Batch Size={}, dropout={}\n'.
                       format(batch, dropout))
            file.write('总训练次数: {}\n'.format(len(model.accuracy)))
            for i in model.accuracy:
                file.write(str(i) + ' ')
            file.write('\n最终准确率: {}\n\n'.format(end_accuracy))

"""
Difference between numpy.array shape (R, 1) and (R,): 
https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r
"""