"""
@Author: wty-yy
@Date: 2023-01-12 11:09:43
@LastEditTime: 2023-01-13 14:55:39
@Description: 根据数据集Simple.txt和Complex.txt绘制人物关系图，并计算输入的人物与Kevin Bacon的关系路径和路径长度.
"""
import time

class Union:  # 并查集
    def __init__(self):
        self.name_id = 0
        self.father = []

    def add_id(self):
        self.father.append(self.name_id)
        self.name_id += 1
        return self.name_id - 1

    def get_father(self, p):  # 查询父节点
        if self.father[p] == p: return p
        self.father[p] = self.get_father(self.father[p])
        return self.father[p]
    
    def join(self, a, b):  # 加入
        self.father[self.get_father(a)] = self.get_father(b)

class Graph:
    def __init__(self):
        self.total_edge = 0  # 记录总边数

    class Node:  # 存储图中节点，每个演员对应一个节点
        def __init__(self, name, id):
            self.id = id
            self.name = name
            self.next = []  # 用邻接表存储后继边

    class Edge():  # 存储图中的边，每个边对应一个电影
        def __init__(self, movie, previous_node, next_node):
            self.movie = movie
            self.previous_node = previous_node
            self.next_node = next_node  # 存储后继节点

    def add_edge(self, node1, node2, movie):  # 加入从node1到node2的单向边
        edge = self.Edge(movie, node1, node2)
        node1.next.append(edge)
        self.total_edge += 1

class Solver():
    def __init__(self, fname, kevin_name):
        self.graph = Graph()
        self.fname = fname
        self.kevin_name = kevin_name
        self.name2node = {}  # 字典存储演员对应的节点

    def read_data(self):
        union = Union()
        with open(self.fname, 'r', encoding='utf-8') as file:  # Bacon, Kevin
            while True:
                string = file.readline()
                if not string: break

                items = string.strip()
                movie = items[0:items.find(')')+1]  # 电影名称
                actors = items[items.find(')')+2:].split('/')  # 演员名称

                for name in actors:  # 创建未见过的节点
                    if name not in self.name2node.keys():
                        name_id = union.add_id()
                        self.name2node[name] = Graph.Node(name, name_id)  # 创建新的节点
                    union.join(self.name2node[actors[0]].id, self.name2node[name].id)

                for i, name1 in enumerate(actors):  # 创建边
                    node1 = self.name2node[name1]
                    for name2 in actors[i+1:]:
                        node2 = self.name2node[name2]
                        self.graph.add_edge(node1, node2, movie)
                        self.graph.add_edge(node2, node1, movie)
            
        self.kevin_node = self.name2node[self.kevin_name]
        self.kevin_id = self.kevin_node.id
        relation_node = 0
        for i in range(union.name_id):  # 计算与Kevin Bacon相关的演员数目
            if union.get_father(i) == union.get_father(self.kevin_id):
                relation_node += 1

        print("总演员数目:", len(self.name2node))
        print("总边数目:", self.graph.total_edge)
        print(f"和{self.kevin_name}相关的演员总数:", relation_node)

    def bfs(self):  # 广度优先搜索
        from queue import Queue
        self.father_edge = {}  # 记录连接到父节点的边
        visited = {self.kevin_node}  # 判断是否访问过该节点
        self.distance = {self.kevin_name: 0}  # 记录最短距离
        q = Queue()
        q.put(self.kevin_node)  # 加入Kevin节点
        while not q.empty():
            u = q.get()
            for e in u.next:
                v = e.next_node  # 访问新的节点
                if v in visited: continue
                visited.add(v)
                self.distance[v.name] = self.distance[u.name] + 1  # 更新距离
                self.father_edge[v] = e  # 记录连接父节点的边
                q.put(v)
    
    def show_path(self, name):  # 回溯打印路径
        if name not in self.distance.keys():  # 若无法到达该节点
            print(f"\nCan't find path from {name} to {self.kevin_name}.")
            return
        print(f"\nPath from {name} to {self.kevin_name}:")
        node = self.name2node[name]
        while node.name != self.kevin_name:  # 利用连接父节点的边，复现路径
            father_edge = self.father_edge[node]
            father_node = father_edge.previous_node
            print(f"{node.name} was in {father_edge.movie} with {father_node.name}")
            node = father_node
        print(f"{name}'s Bacon number is {self.distance[name]}\n")

if __name__ == '__main__':
    solver = Solver("Simple.txt", "Kevin Bacon")
    # solver = Solver("Complex.txt", "Bacon, Kevin")
    start_time = time.time()
    solver.read_data()
    solver.bfs()
    print("预处理用时:", time.time() - start_time, "s")
    print("距离Kevin的最大Bacon距离:", max(solver.distance.values()))
    while True:
        command = input("Actor's name (or All for everyone or Show Bacon distance bigger than NUMBER)?\n> ")
        if command == 'All':
            for name in solver.name2node.keys():
                solver.show_path(name)
        elif 'Show Bacon distance bigger than' in command:  # 新命令，可显示Bacon距离>=某个值的全部节点
            num = int(command.split()[-1])
            for name, distance in solver.distance.items():
                if distance >= num:
                    solver.show_path(name)
        else:
            solver.show_path(command)
        
