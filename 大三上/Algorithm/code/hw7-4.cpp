#include <iostream>
#include <stack>
#include <map>
using namespace std;

// use[][0]表示每个数是否使用过，uses[][1]表示每个符号是否用过
bool use[5][2];
// now[i]表示结果中第i位的值，i为偶数则是数字，反之为运算符
int now[9];
// 存储全部解
map<int, int> mp;

int n, a[5], rk[256];
int opt[] = {'+', '-', '*', '/'};
stack<string> ans;

void execute(stack<int> &stk, int opt) {
    int b = stk.top(); stk.pop();
    int a = stk.top(); stk.pop();
    if (opt == '+') stk.push(a+b);
    if (opt == '-') stk.push(a-b);
    if (opt == '*') stk.push(a*b);
    if (opt == '/') stk.push(a/b);
}

int calc(int x) {  // 利用逆波兰表达式求解当前计算式now[]的值
    stack<int> stk_num, stk_opt;
    for (int i = 0; i < x; i++) {
        if (i % 2 == 0) stk_num.push(now[i]);  // 数字
        else {  // 运算符
            while (!stk_opt.empty() && rk[now[i]] >= rk[stk_opt.top()]) {
                execute(stk_num, stk_opt.top());
                stk_opt.pop();
            }
            stk_opt.push(now[i]);
        }
    }
    while (!stk_opt.empty()) {
        execute(stk_num, stk_opt.top());
        stk_opt.pop();
    }
    mp[stk_num.top()]++;
    return stk_num.top();
}

string vec2str(int x) {  // 将now[]数组转化为字符串
    string ret = "";
    for (int i = 0; i < x; i++) {
        if (i % 2 == 0) ret += to_string(now[i]);
        else ret.push_back(now[i]);
    }
    return ret;
}

void dfs(int x) {  // x为当前枚举位置
    if (x % 2 == 1 && calc(x) == n) {
        string s = vec2str(x);
        ans.push(vec2str(x));
    }
    if (x == 9) return;  // 枚举到头了
    if (x % 2 == 0) {  // 偶数位为数字
        for (int i = 0; i < 5; i++) {
            if (use[i][0]) continue;
            use[i][0] = 1;
            now[x] = a[i];
            dfs(x+1);
            use[i][0] = 0;
        }
    } else {  // 奇数位为运算符
        for (int i = 0; i < 4; i++) {
            if (use[i][1]) continue;
            use[i][1] = 1;
            now[x] = opt[i];
            dfs(x+1);
            use[i][1] = 0;
        }
    }
}

int main() {
    rk['+'] = rk['-'] = 1;  // 设置优先级
    cin >> n;
    for (int i = 0; i < 5; i++) cin >> a[i];
    dfs(0);
    if (ans.size() == 0) cout << "No Solution" << '\n';
    cout << "Total Solution: " << ans.size() << '\n';
    while (!ans.empty()) cout << ans.top() << '\n', ans.pop();

    // 反向求出每个n对应的解的个数
    printf("\nn\tSolution Num\n");
    int tot = 0;
    for (auto i : mp) cout << i.first << '\t' << i.second << '\n', tot += i.second;
    cout << "Total expression: " << tot << '\n';
    return 0;
}
#if 0
Input:
151
1 2 3 10 100
Output:
Total Solution: 6
100/2*3+1
100*3/2+1
3*100/2+1
1+100/2*3
1+100*3/2
1+3*100/2

Input:
666
3 10 20 33 97
Output:
Total Solution: 8
97/10-3+33*20
97/10-3+20*33
97/10+33*20-3
97/10+20*33-3
33*20-3+97/10
33*20+97/10-3
20*33-3+97/10
20*33+97/10-3
#endif