#include <cstdio>
#include <iostream>
#include <utility>
#include <algorithm>
#include <time.h>
#include <string.h>
using namespace std;
const int N = 1e8;

int n, k, a[N], b[N], tmp[N];

void bubble(int l, int r) {  // 对a[l,r)进行冒泡排序
    for (int i = l; i < r; i++)
        for (int j = i + 1; j < r; j++)
            if (a[i] > a[j]) swap(a[i], a[j]);
}
// 划分函数，返回值为：最靠左的mid下标，与mid相同数个数
pair<int, int> partition(int l, int r, int mid) {
    int i = l - 1, j = r;
    while (1) {
        while (a[++i] < mid && i < r);
        while (a[--j] > mid && j >= l);
        if (i >= j) break;
        swap(a[i], a[j]);
    }
    // 例mid=5, [0 3 2 5 5 5 5 14 12 11], ll=2, rr=7
    int ll = j, rr = j+1;  // 与mid相同数合并称一块
    for (int i = 0; i < ll; i++) {
        while (a[ll] == mid && ll > l) ll--;
        if (a[i] == mid) swap(a[i], a[ll]), ll--;
    }
    for (int i = r-1; i > rr; i--) {
        while (a[rr] == mid && rr < r-1) rr++;
        if (a[i] == mid) swap(a[i], a[rr]), rr++;
    }
    return make_pair(ll+1, rr-ll-1);
}

int select(int l, int r, int k) {  // 返回a[l,r)中第k大的元素
    if (r - l < 5) { // 如果元素个数小于5
        bubble(l, r);
        return a[l+k-1];
    }
    // [00000|00000|00000|000]  以五个进行一个划分，每个子区间长度为5
    for (int i = 0; i < (r - l) / 5; i++) {
        int s = l + 5 * i, t = s + 5;  // 处理子区间[s,t)
        for (int j = s; j < s+3; j++)  // 仅需做3次冒泡
            for (int k = j+1; k < t; k++)
                if (a[j] > a[k]) swap(a[j], a[k]);
        swap(a[l+i], a[s+2]);  // 将[s,t)的中位数移动到数列开头
    }
    int x = select(l, l+(r-l)/5, (r-l+5)/10);  // 递归找到中位数的中位数
    auto p = partition(l, r, x);
    int mid = p.first, same = p.second, less = mid-l;  // 以mid作为快排基准进行排序
    if (k <= less) return select(l, mid, k);  // 在左半区间
    else if (k <= less + same) return x;  // 在中间区间就是中位数
    return select(mid+same, r, k-less-same);  // 在右半区间
}

int main1() {  // 用于测试select函数速度
    ios::sync_with_stdio(0);
    cin.tie(0);
    freopen("7-1.in", "r", stdin);
    cin >> n;
    for (int i = 0; i < n; i++) cin >> a[i];
    clock_t start = clock();
    cout << "My Answer: " << select(0, n, 10) << '\n';
    clock_t end = clock();
    cout << "Use Time: " << end - start << "ms" << '\n';
    
    int *tmp = new int[sizeof(a)/sizeof(int)];
    memcpy(tmp, a, sizeof(a));
    start = clock();
    sort(a, a + n);
    cout << "Check: " << a[10-1] << '\n';
    end = clock();
    cout << "Use Time: " << end - start << "ms";
    return 0;
}

int main() {  // 求解距离中位数n/4近的数
    freopen("7-1.in", "r", stdin);
    cin >> n;
    for (int i = 0; i < n; i++) cin >> a[i];
    memcpy(tmp, a, sizeof(int) * n);
    int mid = select(0, n, (n+1)/2);  // 先求出中位数
    cout << "mid: " << mid << '\n';
    for (int i = 0; i < n; i++) a[i] = abs(a[i] - mid);  // 求出绝对值数组
    int k = select(0, n, n/4), tot = n/4;  // 绝对值数组中前n/4分位数
    memcpy(a, tmp, sizeof(int) * n);
    cout << "Around Mid: ";
    for (int i = 0; i < n && tot; i++)  // 再从原数组中找绝对值差小于等于k的
        if (abs(a[i] - mid) <= k) {
            cout << a[i] << ' ';
            tot--;
        }
    cout << '\n';

    cout << "\n" << "After sorted(Check): " << '\n';
    sort(a, a + n);
    for (int i = 0; i < n; i++)
        cout << a[i] << ' ';
    cout << '\n';
    return 0;
}

#if 0
Input:
9
1 2 2 3 4 5 5 8 9
Output:
mid: 4
Around Mid: 3 4

Input:
20
34 26 98 31 26 88 90 39 68 95 80 78 69 7 3 48 32 39 9 63 
Output:
mid: 39
Around Mid: 34 31 39 32 39
#endif