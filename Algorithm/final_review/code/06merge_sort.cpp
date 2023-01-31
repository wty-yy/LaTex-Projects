/*
 * @Author: wty-yy
 * @Date: 2023-01-26 17:01:16
 * @LastEditTime: 2023-01-26 17:17:28
 * @Description: 合并排序
 */
#include <iostream>
#include <cstring>
using namespace std;

const int N = 1e5;
int b[N];  // 临时数组
void merge(int a[], int left, int mid, int right) {
    int i = left, j = mid, k = 0;
    while (i < mid && j < right) {
        if (a[i] < a[j]) b[k++] = a[i++];
        else b[k++] = a[j++];
    }
    while (i < mid) b[k++] = a[i++];
    while (j < right) b[k++] = a[j++];
}
void merge_sort(int a[], int left, int right) {
    if (left + 1 == right) return;
    int mid = (left+right) / 2;
    merge_sort(a, left, mid);
    merge_sort(a, mid, right);
    merge(a, left, mid, right);
    memcpy(a+left, b, (right - left) * sizeof(int));
}

int main() {
    int a[] = {5, 9, 2, 4, 1, 3}, n = sizeof(a)/sizeof(int);
    merge_sort(a, 0, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    // Output: 1 2 3 4 5 9
    return 0;
}