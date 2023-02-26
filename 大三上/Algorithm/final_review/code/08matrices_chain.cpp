/*
 * @Author: wty-yy
 * @Date: 2023-01-27 09:09:39
 * @LastEditTime: 2023-01-27 09:27:34
 * @Description: 求解矩阵连乘所需最少运算次数，通过加入括号改变运算次数
 * 使用dp求解，类似石子合并问题
 */
#include <iostream>
#include <cstring>
using namespace std;

const int N = 1e3;
const int INF = 0x3f3f3f3f;
int n = 6;
int dp[N][N], row[N] = {30, 35, 15, 5, 10, 20, 25};
int solve(int l, int r) {
    if (dp[l][r] != INF) return dp[l][r];
    for (int k = l+1; k < r; k++)
        dp[l][r] = min(dp[l][r], solve(l, k)+solve(k, r)+row[l]*row[k]*row[r]);
    return dp[l][r];
}
int main() {
    memset(dp, 0x3f, sizeof(dp));  // 初始化为极大值INF
    for (int i = 0; i < n; i++) dp[i][i+1] = 0;
    printf("%d\n", solve(0, n));  // Output: 15125
    return 0;
}