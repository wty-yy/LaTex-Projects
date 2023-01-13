/*
 * @Author: wty-yy
 * @Date: 2023-01-12 15:42:00
 * @LastEditTime: 2023-01-12 18:52:20
 * @Description: 
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <time.h>
using namespace std;

struct Edge;

struct Node {  // 存储节点信息
    string name;  // 演员名称
    vector<Edge*> edge;  // 记录出边
    Node(string name):name(name){}
};

const int M = 2e7;
int edge_count, movie_id;
map<int, string> id2movie;
map<string, int> movie2id;
struct Edge {  // 存储边信息
    int movie_id;  // 电影名称
    Node* to;  // 出边到的节点
} e[M];

int total_edges = 0;
void add_edge(Node* a, Node* b, int movie_id) {  // 连接节点a到节点b
    Edge* edge = &e[edge_count++];
    edge->movie_id = movie_id;
    edge->to = b;
    a->edge.push_back(edge);
    total_edges += 1;
}

map<string, Node*> name2node;
int main() {
    time_t start_time = clock();
    freopen("Complex.txt", "r", stdin);
    string s;
    while (getline(cin, s)){
        string movie;
        vector<string> actors;
        for (int l = 0, r = 0; l <= s.size(); l = r + 1) {
            while (r < s.size() && s[++r] != '/');
            if (!l) movie = s.substr(l, r - l);
            else actors.push_back(s.substr(l, r - l));
        }
        if (!movie2id.count(movie)) {
            movie2id[movie] = movie_id;
            id2movie[movie_id++] = movie;
        }

        for (string name : actors)
            if (!name2node.count(name))
                name2node[name] = new Node(name);

        for (int i = 0; i < actors.size(); i++) {
            for (int j = i + 1; j < actors.size(); j++) {
                add_edge(name2node[actors[i]], name2node[actors[j]], movie2id[movie]);
                add_edge(name2node[actors[j]], name2node[actors[i]], movie2id[movie]);
            }
        }
        // break;
    }
    cout << "总人数:" << name2node.size() << '\n';
    cout << "边数:" << total_edges << '\n';
    cout << "电影数:" << movie2id.size() << '\n';
    cout << "用时:" << 1.0 * (clock() - start_time) / CLOCKS_PER_SEC << '\n';
    return 0;
}