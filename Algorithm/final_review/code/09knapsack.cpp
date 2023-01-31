/*
 * @Author: wty-yy
 * @Date: 2023-01-29 17:58:15
 * @LastEditTime: 2023-01-29 18:13:48
 * @Description: 01背包，模板题 https://www.acwing.com/problem/content/2/
 */
#include <iostream>
using namespace std;

const int N = 1001;
int dp[N][N];
int knapsack(int n, int c, int v[], int w[]) {
    for (int j = w[n]; j <= c; j++) dp[n][j] = v[n];  // 初始化边界条件
    for (int i = n-1; i >= 1; i--) {
        for (int j = 0; j <= c; j++) {
            dp[i][j] = dp[i+1][j];  // 从上个状态转移过来
            if (j >= w[i])  // 考虑是否加入第i个物品
                dp[i][j] = max(dp[i][j], v[i] + dp[i+1][j-w[i]]);
        }
    }
    return dp[1][c];
}

int v[N], w[N];
int main() {
    int n, c;
    scanf("%d %d", &n, &c);
    for (int i = 1; i <= n; i++) scanf("%d %d", &w[i], &v[i]);
    printf("%d", knapsack(n, c, v, w));
    return 0;
}