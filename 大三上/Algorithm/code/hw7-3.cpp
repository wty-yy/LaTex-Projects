#include <iostream>
#include <utility>
#include <algorithm>
using namespace std;

const int N = 1e6 + 10;
int n;
pair<int, int> a[N];  // 利用pair存储点对坐标

int main() {
    freopen("7-3.in", "r", stdin);
    cin >> n;
    for (int i = 0; i < n; i++) cin >> a[i].first >> a[i].second;
    sort(a, a+n);  // pair二元组自带比较关系，优先x从小到大
    int mx = 0;  // 记录最大y值
    cout << "maxima: " << '\n';
    for (int i = n-1; i >= 0; i--) {
        if (a[i].second < mx) continue;
        mx = max(mx, a[i].second);
        cout << a[i].first << ' ' << a[i].second << '\n';
    }
    return 0;
}
#if 0
Input:
10
8 7
6 4
9 5
1 2
5 1
0 0
4 9
3 8
2 6
7 3
Output:
9 5
8 7
4 9
#endif