import numpy as np
from timer import Timer


class Function:  # 存储各种函数及其导数
    @staticmethod
    def sigmoid(x, order=0):
        """
        sigmoid(x) = 1/(1+exp(-x))
        Args:
            x: 输入变量
            order: 导数的阶数
        Returns:
            1/(1+exp(-x))
        """
        assert order < 2  # 只处理0和1阶导
        if order == 0:
            return 1 / (1 + np.exp(-x))
        return Function.sigmoid(x) * (1 - Function.sigmoid(x))

    @staticmethod
    def tanh(x, order=0):
        """
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        """
        assert order < 2
        if order == 0:
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return 1 - np.power(Function.tanh(x), 2)

    @staticmethod
    def ReLU(x, order=0):
        """
        ReLU(x) = max(0, x)
        """
        assert order < 2
        if isinstance(x, int):
            if order == 0:
                return max(x, 0)
            if x > 0:
                return 1
            return 0
        n = x.shape[0]
        if order == 0:
            for i in range(n):
                x[i] = max(x[i], 0)
            return x
        for i in range(n):
            if x[i] > 0:
                x[i] = 1
        return x

    @staticmethod
    def sse(x, y=0, order=0):
        """
        平方损失函数 Quadratic Loss Function
        误差平方和 Sum of Squared Errors
        sse = sum((x-y)^2)
        """
        assert order < 2
        if order == 0:
            return np.sum((x - y) ** 2)
        return 2 * (x - y)

    @staticmethod
    def cross_entropy(x, y=0, order=0):
        """
        交叉熵损失函数 Cross-Entropy Loss Function
        -log(x[y])
        """
        assert order < 2
        n = y.shape[0]
        c = 0  # 样本的标签
        for i in range(n):
            if y[i] == 1:
                c = i
        x = x / np.sum(x)  # 转成概率分布形式
        if order == 0:
            return -np.log(x[c])
        ret = np.zeros(n).reshape(n, 1)
        ret[c] = -1 / x[c]
        return ret

# 参考下面文章, 利用字典将字符串和静态函数相对应
# https://stackoverflow.com/questions/41921255/staticmethod-object-is-not-callable
Function.switch = {
    'sigmoid': Function.sigmoid,
    'tanh': Function.tanh,
    'ReLU': Function.ReLU,
    'sse': Function.sse,
    'cross_entropy': Function.cross_entropy,
}


class Layer:  # 神经网络中的每一层
    n, m = None, None  # n为输入维数, m为输出维数
    weight, bias = None, None  # 权重矩阵, 偏置向量
    grad_weight, grad_bias = None, None  # 权重梯度, 偏置梯度
    activation, regular = None, None  # 激活函数, 正则化方法
    R = None  # 活跃矩阵, 与dropout参数相关, 计算每一层所使用的神经元

    def __init__(self, n, m, activation='sigmoid', dropout=0):
        self.n, self.m = n, m
        self.activation = Function.switch[activation]
        # 初始化偏置向量和权重矩阵
        self.bias = np.zeros((m, 1))
        self.weight = np.random.rand(m, n) - 0.5
        self.dropout = dropout  # 设定抛弃率

    def random_dropout(self):
        self.R = np.random.binomial(1, 1-self.dropout, self.m) * np.identity(self.m)  # 初始化活跃矩阵

    def forward(self, x_input):  # 向前传播, 计算当前层的激活值
        z = self.R @ (self.weight @ x_input + self.bias)
        return z, self.activation(z)

    def back(self, a, z, delta):  # 反向传播
        """
        设当前层为l, 计算当前层的偏置和权重的梯度, 并求出l-1层的delta值
        Args:
            a: l-1层的激活值
            z: l-1层的加权值 z = w*a+b
            delta: l层的delta值
        Returns:
            delta, (np.array, list): l-1层的delta值和权重梯度和偏置梯度展成行向量的结果
        """
        grad_weight = self.R @ delta @ a.T
        grad_bias = self.R @ delta
        return self.activation(z, order=1) * (self.weight.T @ (self.R @ delta)), [grad_weight, grad_bias]


class Model:  # 神经网络模型
    tot_input, output = None, None

    def __init__(self, *args, mu=1, loss='sse', test_input=None, test_output=None,
                 test_rate=0.1):
        self.layers = list(args)  # 神经网络结构
        self.mu = mu  # 学习率
        self.loss = Function.switch[loss]  # 损失函数
        self.test_input, self.test_output = test_input, test_output  # 测试集用于计算准确率, 不用于训练
        self.memory = False  # 判断是否需要记录, 每次更新梯度次数时的准确率
        self.accuracy = []  # 用于存储准确率
        self.test_rate = test_rate  # 如果存储每次梯度更新后的准确度, 计算准确度所需的测试集大小占比

    def add(self, *args):  # 在末尾继续增加一层
        self.layers.append(args)

    def propagation(self, x_input, output=None):
        """
        完成一次传播（包括前向传播和反向传播计算梯度）, 如果output没有输入则只用于做预测
        Args:
            x_input: 输出参数
            output: 期望输出
        Returns:
            总梯度
        """
        grad = []  # 用list存储总梯度
        L = len(self.layers)  # 网络总层数
        a = []  # 向前传播中, 每一层的激活值, 加权值
        n = np.size(x_input)
        x_input = x_input.reshape(n, 1)
        a.append((x_input, 0))
        for i in range(L):  # 向前传播
            layer = self.layers[i]
            layer.random_dropout()  # 每次训练随机产生活跃矩阵
            a.append(layer.forward(a[i][0]))
        y, layer = a[L][1], self.layers[L - 1]  # 预测值y, 最后一层layer
        new_output = (y == np.max(y)).astype(int)  # 当前正规化预测值
        if output is None:
            return new_output
        m = np.size(output)
        output = output.reshape(m, 1)
        loss = self.loss(y, output)
        delta = layer.activation(a[L][0], order=1) * self.loss(y, output, order=1)  # 初始化误差值
        for i in range(L-1, -1, -1):  # 反向传播, 计算梯度
            layer = self.layers[i]  # 倒着取值
            delta, g = layer.back(*a[i], delta)  # 计算每一层的梯度
            grad = g + grad
        return grad, new_output, loss

    def batch(self, arr, length):
        """
        计算每一个Mini-batch, 并对网络参数进行调整
        Args:
            arr: 用list存储当前batch中的训练编号
            length: arr的长度
        Returns:
            平均误差
        """
        L = len(self.layers)  # 总层数
        grad, loss = None, None  # 当前batch的平均梯度, 和loss的平均值
        for i in arr:  # 开始训练(此处可以加入多线程)
            x_input = np.asarray(self.tot_input[i]).T
            output = self.output[i]
            g, new_output, l = self.propagation(x_input, output)
            if grad is None:
                grad = g
                loss = l
            else:
                for j in range(2 * L):
                    grad[j] = grad[j] + g[j]  # 对每个数据的梯度进行累加
                loss += l
        for i in range(L):  # 更新梯度
            j = i * 2
            grad[j] = grad[j] / length  # 对权重梯度求平均值
            self.layers[i].weight = self.layers[i].weight - self.mu * grad[j]
            grad[j + 1] = grad[j + 1] / length  # 对偏置梯度求平均值
            self.layers[i].bias = self.layers[i].bias - self.mu * grad[j + 1]
        return loss / length  # 返回平均误差

    def fit(self, tot_input, output, epoch=5, batch=1, memory=False, tot_update=None):
        """
        Args:
            tot_input: 总的输入训练数据
            output: 总的输出训练数据
            epoch: 训练数据集次数
            batch: 每个Mini-batch的大小
            memory: 是否需要记录准确值
            tot_update: 总梯度更新次数(如果不为None, 则epoch参数无效, 一直迭代直到梯度更新次数达到tot_update)
        """
        update_counter = 0
        if tot_update is not None:  # 如果总梯度更新次数不为None
            epoch = int(1e9)  # 设置epoch为最大值
        self.memory = memory
        timer = Timer()
        self.tot_input, self.output = tot_input, output
        n, m = np.shape(tot_input)  # n为总数据量, m为输入数据向量的维数
        B = n // batch  # Mini-batch的总个数
        for _ in range(epoch):
            rand = np.random.permutation(n)
            for i in range(B):
                st, en = i * batch, (i + 1) * batch  # 划分数据集
                loss = self.batch(rand[st:en], en - st)  # 用每个batch对网络参数进行调整
                update_counter += 1
                if update_counter == tot_update:
                    break
            if self.memory:
                self.accuracy.append(self.evaluate())
                # print('accuracy: {}'.format(self.accuracy[-1]))
            print('epoch {}, 准确率: {}, '.format(_+1, self.accuracy[-1]), end='')
            timer.show()
            if update_counter == tot_update:
                break

    def evaluate(self, test_rate=None):
        """
        评估模型的准确率
        Args:
            test_rate: 多大比例的测试样本(随机选取),
            如果传入是None默认使用self.test_rate, 计算每次更新梯度后的准确率
        Returns:
            准确率
        """
        if test_rate is None:
            test_rate = self.test_rate
        correct = 0
        n, m = np.shape(self.test_input)
        rand = np.random.permutation(n)  # 打乱测试样本
        n = int(n * test_rate)  # 测试样本个数
        for i in rand[:n]:
            a_input = self.test_input[i].reshape(m, 1)
            new_output = self.propagation(a_input)
            if list(self.test_output[i]) == list(new_output):
                correct += 1
        return correct / n


if __name__ == '__main__':
    pass
