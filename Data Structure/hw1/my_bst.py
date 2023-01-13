class BSTBase():  # BST基类
    def insert(self, key, value): pass  # 插入(key,value)键值对
    def remove(self, key): pass  # 删除key节点，若找到并删除节点，则返回对应的value，若无该节点则返回None
    def search(self, key): pass  # 查询key对应的value，若无key节点，则返回None
    def update(self, key, value): pass  # 更新key对应的value，若无key节点，返回False，否则返回True
    def isEmpty(self): pass  # 判断二叉搜索树是否为空
    def clear(self): pass  # 重新初始化BST
    def showStructure(self, file): pass  # 输出当前二叉树的节点总数和高度到文件file中
    def printInorder(self, file): pass  # 输出二叉树的中序遍历到文件file中

class BST(BSTBase):
    class Node():  # 节点子类
        val0 = 0
        def __init__(self):
            self.key = None  # 初始化键值
            self.child = [None, None]  # 初始化左右孩子节点
            if isinstance(self.val0, list):
                self.val = []  # 由于list按照实参赋值，必须重新创建空list
            else: self.val = self.val0

    def __init__(self, val0=0):
        self.Node.val0 = val0  # val0为每个节点值的初值
        self.file = None  # 将要写入的文件
        self.root = None  # 创建根节点
        self.height = 0  # 树的高度
        self.size = 0  # 树的节点总数

    def add(self, p, key, val):
        if p is None:  # 当前节点为空节点, 开始创建
            p = self.Node()
            p.key = key
        if key == p.key:  # 找到key值对应节点, 更新节点val
            if isinstance(p.val, list):  # 如果val当前是列表, 则加入值
                p.val.append(val)
            else:  # 否则直接修改当前值
                p.val = val
        elif key < p.key:  # key节点在左子树中
            p.child[0] = self.add(p.child[0], key, val)
        else:  # key节点在右子树中
            p.child[1] = self.add(p.child[1], key, val)
        return p

    def insert(self, key, val):
        if key is None or val is None:
            return
        self.root = self.add(self.root, key, val)  # 每次从根节点开始查找插入位置

    def delete_min(self, p):  # 查找最小值
        if p.child[0] is None:  # 找到最小值
            return p, p.child[1]  # 返回的第一个参数为找到的最小值点, 第二个参数为更新点编号
        min_p, update_p = self.delete_min(p.child[0])
        p.child[0] = update_p  # 用第二个参数更新该点
        return min_p, p  # 返回的第一个参数为找到的最小值点, 第二个参数为更新点编号

    def delete(self, p, key):
        if p.key == key:
            if p.child[0] and p.child[1]:  # 如果有两个儿子节点, 就需要找到左子树中的最大值或右子树的最小值替代
                # 这里找右子树最小值替代
                min_p, update_p = self.delete_min(p.child[1])
                min_p.child[0] = p.child[0]
                min_p.child[1] = update_p
                del p  # 删除掉当前节点p
                return min_p
            else:
                ret = p.child[0] if p.child[0] else p.child[1]
                del p  # 删除掉当前节点p
                return ret
        elif key < p.key:  # key节点在左子树中
            p.child[0] = self.delete(p.child[0], key)
        else:  # key节点在右子树中
            p.child[1] = self.delete(p.child[1], key)
        return p

    def remove(self, key):
        val = self.search(key)
        if key is None or val is None:  # 如果找不到该key值也返回False
            return None
        self.root = self.delete(self.root, key)
        return val

    def find(self, p, key):
        if p == None:  # 空节点
            return None
        elif key == p.key:  # 找到了key节点
            return p.val
        elif key < p.key:  # key节点在左子树中
            return self.find(p.child[0], key)
        return self.find(p.child[1], key)  # 否则只能在右子树中

    def search(self, key):
        if key is None:
            return None
        return self.find(self.root, key)

    def update(self, key, val):
        if key is None or self.search(key) is None:  # 如果找不到该key值也返回False
            return False
        self.insert(key, val)  # 用insert修改相同key节点的val值
        return True

    def isEmpty(self):
        return self.root == None

    def clear(self):
        self.dfs(self.root, delete_node=True)
        self.__init__()  # 调用初始化函数, 清空BST

    def struct(self, p, h):  # 查找bst的节点数和高度
        self.size += 1  # 总节点数+1
        self.height = max(self.height, h)  # 更新树的深度
        if p.child[0]:
            self.struct(p.child[0], h + 1)
        if p.child[1]:
            self.struct(p.child[1], h + 1)

    def showStructure(self, file):  # 返回树的总节点数, 返回树的高度
        if file is None:
            return None
        self.size, self.height = 0, 0  # 先初始化为0
        self.struct(self.root, 1)
        file.write('-----------------------------\n')
        file.write(f'There are {self.size} nodes in this BST.\nThe height of this BST is {self.height}.\n')
        file.write('-----------------------------\n')

    def dfs(self, p, delete_node=False):  # 输出中序遍历结果
        if p.child[0]:
            self.dfs(p.child[0], delete_node)
        if not delete_node:
            self.file.write('[{} ---- < {} >]\n'.format(p.key, str(p.val)[1:-1]))  # 在文件中直接写入
        if p.child[1]:
            self.dfs(p.child[1], delete_node)
        if delete_node:  # 用于清空整棵树
            del p

    def printInorder(self, file):
        if file is None:
            return None
        self.file = file
        self.dfs(self.root)
        return   # 返回中序遍历
