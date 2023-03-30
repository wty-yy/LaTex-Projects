import numpy as np

class Layer:  # 神经网络中的每一层
    """
    Initialize
        当前层和上一层的权重矩阵: weight
        当期层的偏置: bias
        当前层所用的激活函数: activation

    """
    def forward(self, x_input):  # 向前传播，计算当前层的激活值
        z = self.weight @ x_input + self.bias
        return z, self.activation(z)

    def back(self, a, z, delta):  # 反向传播
        """
        设当前层为l，计算当前层的偏置和权重的梯度，并求出l-1层的delta值
        Input
            a: l-1层的激活值
            z: l-1层的加权值 z = w*a+b
            delta: l层的delta值
        Output
            np.array, list: l-1层的delta值和权重梯度和偏置梯度展成行向量的结果
        """
        grad_weight = (delta @ a.T)
        grad_bias = delta
        return self.activation(z, order=1) * (self.weight.T @ delta), [grad_weight, grad_bias]

class Model:  # 神经网络模型
    """
    Initialize
        layers: 由Layer类为元素的list
    """
    def propagation(self, x_input, output=None):
        """
        完成一次传播（包括前向传播和反向传播计算梯度），如果output没有输入则只用于做预测
        Input
            x_input: 输出参数
            output: 期望输出
        Output
            总梯度
        """
        grad = []  # 用list存储总梯度
        L = len(self.layers)  # 网络总层数
        a = []  # 向前传播中，每一层的激活值，加权值
        n = np.size(x_input)
        x_input = x_input.reshape(n, 1)
        a.append((x_input, 0))
        for i in range(L):  # 前向传播
            layer = self.layers[i]
            a.append(layer.forward(a[i][0]))
        y, layer = a[L][1], self.layers[L - 1]  # 预测值y，最后一层layer
        new_output = (y == np.max(y)).astype(int)  # 当前正规化预测值
        if output is None:
            return new_output
        m = np.size(output)
        output = output.reshape(m, 1)
        loss = self.loss(y, output)
        delta = layer.activation(a[L][0], order=1) * self.loss(y, output, order=1)  # 初始化误差值
        for i in range(L-1, -1, -1):  # 反向传播，计算梯度
            layer = self.layers[i]  # 倒着取值
            delta, g = layer.back(*a[i], delta)  # 计算每一层的梯度
            grad = g + grad
        return grad, new_output, loss
