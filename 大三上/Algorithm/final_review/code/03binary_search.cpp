/*
 * @Author: wty-yy
 * @Date: 2023-01-26 10:52:05
 * @LastEditTime: 2023-01-26 11:07:29
 * @Description: 二分搜索
 */
#include <iostream>
#include <algorithm>
using namespace std;

// 返回第一个的大于等于x的索引
int binary_search(int a[], int x, int n) {
    int l = 0, r = n-1;
    while (l < r) {
        int mid = (l + r) / 2;
        if (x > a[mid]) l = mid + 1;
        else r = mid - 1;
    }
    return r+1;
}
int main() {
    int a[] = {1, 2, 5, 8, 10}, x = 5, n = sizeof(a)/sizeof(int);
    printf("%d=a[%d]\n", x, binary_search(a, x, n));
    // 与algorithm库中lower_bound函数功能一致
    printf("%d=a[%d]", x, lower_bound(a, a+n, x)-a);
    return 0;
}