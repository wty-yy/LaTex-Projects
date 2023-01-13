class BST():
    # 初始化方法, val0为val数组的初值
    def __init__(self, max_size=1000000, val0=0):
        self.file = None  # 将要写入的文件
        self.n = 1  # 记录当前所有节点的最大编号+1
        self.root = 0  # 记录根节点编号
        self.height = 0  # 树的高度
        self.size = 0  # 树的节点总数
        self.max_size = max_size  # max_size为bst的最大容量
        if isinstance(val0, list):
            self.val = [[] for _ in range(max_size)]  # 用空列表进行初始化
        else:
            self.val = [val0 for _ in range(max_size)]  # 用val进行初始化
        self.key, self.lc, self.rc = ([0] * max_size for _ in range(3))  # 初始化列表

    def add(self, p, key, value):
        if self.key[p] == 0:  # 当前节点为空节点, 开始创建
            p = self.n
            self.key[p] = key
            self.n += 1
        if key == self.key[p]:  # 找到当前节点, 更新节点value
            if isinstance(self.val[p], list):  # 如果val当前是列表, 则加入值
                self.val[p].append(value)
            else:  # 否则直接修改当前值
                self.val[p] = value
        elif key < self.key[p]:  # key节点在左子树中
            self.lc[p] = self.add(self.lc[p], key, value)  # 构建子节点的同时更新当前节点
        else:  # key节点在右子树中
            self.rc[p] = self.add(self.rc[p], key, value)  # 构建子节点的同时更新当前节点
        return p

    def insert(self, key, value):
        if key is None or value is None or self.n == self.max_size:
            return False
        self.root = self.add(self.root, key, value)  # 每次从根节点开始查找插入位置
        return True

    def delete_min(self, p):  # 查找最小值
        if self.lc[p] == 0:  # 找到最小值
            return p, self.rc[p]  # 返回的第一个参数为找到的最小值点, 第二个参数为更新点编号
        tmp = self.delete_min(self.lc[p])
        self.lc[p] = tmp[1]  # 用第二个参数更新该点
        return tmp[0], p  # 返回的第一个参数为找到的最小值点, 第二个参数为更新点编号

    def delete(self, p, key):
        if self.key[p] == key:
            self.key[p] = 0  # 删除节点, 将该节点的key值清空, 用于判断是否删除
            if self.lc[p] and self.rc[p]:  # 如果有两个儿子节点, 就需要找到左子树中的最大值或右子树的最小值替代
                # 这里找右子树最小值替代, 也可试试用左子树最大值替代
                tmp = self.delete_min(self.rc[p])
                self.rc[p] = tmp[1]
                self.lc[tmp[0]] = self.lc[p]
                self.rc[tmp[0]] = self.rc[p]
                return tmp[0]  # 寻找右子树中的最小值作为替代值
            else:  # 否则为叶子节点, 或者为链节点(只有一个儿子节点)
                return self.lc[p] + self.rc[p]
        elif key < self.key[p]:  # key节点在左子树中
            self.lc[p] = self.delete(self.lc[p], key)
        else:  # key节点在右子树中
            self.rc[p] = self.delete(self.rc[p], key)
        return p

    def remove(self, key):
        val = self.search(key)
        if key is None or not val:  # 如果找不到该key值也返回False
            return False
        self.root = self.delete(self.root, key)
        return val

    def find(self, p, key):
        if self.key[p] == 0:  # 空节点
            return False
        elif key == self.key[p]:  # 找到了key节点
            return self.val[p]
        elif key < self.key[p]:  # key节点在左子树中
            return self.find(self.lc[p], key)
        return self.find(self.rc[p], key)  # 否则只能在右子树中

    def search(self, key):
        if key is None:
            return False
        return self.find(self.root, key)

    def update(self, key, value):
        if key is None or not self.search(key):  # 如果找不到该key值也返回False
            return False
        self.insert(key, value)  # 用insert修改相同key节点的value值
        return value

    def isEmpty(self):
        return self.key[self.root] == 0

    def clear(self):
        self.__init__()  # 调用初始化函数, 清空BST

    def struct(self, p, h):  # 查找bst的节点数和高度
        self.size += 1  # 总节点数+1
        self.height = max(self.height, h)  # 更新树的深度
        if self.lc[p]:
            self.struct(self.lc[p], h + 1)
        if self.rc[p]:
            self.struct(self.rc[p], h + 1)

    def showStructure(self, outputFile):  # 返回树的总节点数, 返回树的高度
        if outputFile is None:
            return None
        self.size, self.height = 0, 0  # 先初始化为0
        self.struct(self.root, 1)
        return self.size, self.height

    def dfs(self, p):  # 输出中序遍历结果
        if self.lc[p]:
            self.dfs(self.lc[p])
        self.file.write('[{} ---- < {} >]\n'.format(self.key[p], str(self.val[p])[1:-1]))  # 在文件中直接写入
        if self.rc[p]:
            self.dfs(self.rc[p])

    def printInorder(self, outputFile):
        if outputFile is None:
            return None
        self.file = open(outputFile, 'w')  # 打开文件
        self.dfs(self.root)
        self.file.close()  # 记得关闭
        return   # 返回中序遍历
