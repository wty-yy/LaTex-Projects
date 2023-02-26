#include <bits/stdc++.h>
using namespace std;
int a[] = {1,2,3,4,5};  // 中位数为3.5
int b[] = {3,4,5,6,7};
// int a[] = {1,2,3,4,5};  // 中位数为5.5
// int b[] = {6,7,8,9,10};
// int a[] = {3,4,5,6,7};  // 中位数为3.5
// int b[] = {1,2,3,4,5};

double calc_mid(int *a, int l, int r) { // 计算数组a[l~r]的中位数
    int k = (l + r) / 2;
    if ((r - l + 1) % 2 == 0) {
        return (a[k] + a[k+1]) / 2.0;
    } else {
        return a[k];
    }
}
double get_mid(int l1, int r1, int l2, int r2) {
    if (l1 == r1-1) {  // 枚举到端点，中位数在两个数组之间
        return (a[l1] + b[l2]) / 2.0;
    }
    double mid1 = calc_mid(a, l1, r1);
    double mid2 = calc_mid(b, l2, r2);
    if (abs(mid1 - mid2) < 1e-6) {
        return mid1;
    }
    if (mid1 < mid2) {
        return get_mid((l1+r1)/2, r1, l2, (l2+r2)/2);
    } else return get_mid(l1, (l1+r1)/2, (l2+r2)/2, r2);
}

int main() {
    int n = sizeof(a) / sizeof(int);
    printf("中位数为%.2f\n", get_mid(0, n, 0, n));
	return 0;
}
