/*
 * @Author: wty-yy
 * @Date: 2023-01-26 17:28:59
 * @LastEditTime: 2023-01-26 17:59:27
 * @Description: 快速排序
 */
#include <iostream>
#include <time.h>
using namespace std;

int random_partition(int a[], int left, int right) {
    int base_index = rand() % (right - left) + left;
    int x = a[base_index];
    swap(a[left], a[base_index]);
    int i = left+1, j = right-1;
    while (true) {
        while (a[i] <= x && i < right) i++;
        while (a[j] >= x && j > left) j--;
        if (i >= j) break;
        swap(a[i], a[j]);
    }
    swap(a[left], a[j]);
    return j;
}
void quick_sort(int a[], int left, int right) {
    if (left + 1 >= right) return;
    int base_index = random_partition(a, left, right);
    quick_sort(a, left, base_index);
    quick_sort(a, base_index+1, right);
}

int main() {
    srand(time(NULL));  // 根据时间初始化随机种子
    int a[] = {5, 9, 2, 4, 1, 3}, n = sizeof(a)/sizeof(int);
    quick_sort(a, 0, n);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    // Output: 1 2 3 4 5 9
    return 0;
}