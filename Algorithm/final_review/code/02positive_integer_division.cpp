/*
 * @Author: wty-yy
 * @Date: 2023-01-25 14:02:39
 * @LastEditTime: 2023-01-25 14:12:50
 * @Description: 求解正整数划分的个数，例如4有以下5种划分.
 * 4 = 3+1 = 2+2 = 2+1+1 = 1+1+1+1
 */
#include <iostream>
using namespace std;

int f(int n, int m) {
    if (n == 0 || n == 1 || m == 1) return 1;
    if (n < m) return f(n, n);
    return f(n-m, m) + f(n, m-1);
}

int main() {
    int n;
    cin >> n;
    cout << f(n, n);
    return 0;
}