#include <cstdio>
#include <vector>
using namespace std;
int a[] = {1, 1, 5, 5, 1, 5, 1};  // 有主元
// int a[] = {1, 1, 5, 5, 1, 5};  // 没有主元
// int a[] = {1, 1, 5, 5, 1, 1};  // 有主元
// int a[] = {1, 3, 2, 5, 1, 5, 1};  // 没有主元
bool check(int x, int y) {
    // 这个就是交互用的，专门用于返回a[x],a[y]是否相等，
    // 其他函数只能通过该函数访问数组
    return a[x] == a[y];
}
// v存储当前可能是主元的下标
bool find(vector<int> v) {  // 只能通过下标数组判断
    int n = v.size();
    if (n == 0) {
        return false;
    }
    vector<int> Q;
    for (int i = 1; i < n; i += 2) {
        if (check(v[i-1], v[i])) {
            Q.push_back(v[i]);
        }
    }
    if (n % 2 == 1) {
        int cnt = 1;
        for (int i = 0; i < n-1; i++) {
            if (check(v[i], v[n-1])) {
                cnt++;
            }
        }
        if (cnt > n/2) {
            return true;
        }
    }
    return find(Q);
}
int main() {
	int n = sizeof(a) / sizeof(int);
	vector<int> v;  // 需要判断的下标数组
	for (int i = 0; i < n; i++) {
        v.push_back(i);
	}
	if (find(v)) {
        printf("有主元\n");
	} else {
	    printf("没有主元\n");
	}
}

