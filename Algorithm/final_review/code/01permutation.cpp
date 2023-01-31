/*
 * @Author: wty-yy
 * @Date: 2023-01-25 10:26:33
 * @LastEditTime: 2023-01-25 11:11:22
 * @Description: 实现递归和非递归的数字全排列
 */

#include <iostream>
#include <cstring>
#include <time.h>
using namespace std;
int n = 9;
char a[100000];

void perm(char a[], int i, int n, bool verbose=0) {
    if (i == n) {
        if (verbose)
            printf("%s\n", a);
        return;
    }
    for (int j = i; j < n; j++) {
        swap(a[i], a[j]);
        perm(a, i+1, n);
        swap(a[i], a[j]);
    }
}

void perm_non_recursion(char a[], int n, bool verbose=0) {
    bool *flag = new bool[n];
    int *index = new int[n];
    memset(flag, n, 0);
    int i = 0, t = 0;
    while (i >= 0) {
        if (i == n) {
            if (verbose) {
                for (int j = 0; j < n; j++) putchar(a[index[j]]);
                putchar('\n');
            }
            t = n;
        }
        while (t < n && flag[t]) t++;
        if (t < n) {
            flag[t] = 1;
            index[i++] = t;
            t = 0;
        } else {
            t = index[--i];
            if (i >= 0)
                flag[t++] = 0;
        }
    }
}

int main() {
    clock_t start = clock();
    for (int i = 0; i < n; i++) a[i] = '0' + i;
    perm(a, 0, n);
    double perm_time = 1.0*(clock() - start) / CLOCKS_PER_SEC;
    start = clock();
    perm_non_recursion(a, n);
    double perm_non_recursion_time = 1.0*(clock() - start) / CLOCKS_PER_SEC;
    printf("递归: %.2fs\n非递归: %.2fs", perm_time, perm_non_recursion_time);
    return 0;
}