#include <iostream>
using namespace std;
const int N = 1001;
int w[N], v[N];
int dp[N][N];
bool use[N][N];
int main() {
    int tot, n;
    cin >> tot >> n;
    for (int i = 1; i <= n; i++) cin >> w[i];
    for (int i = 1; i <= n; i++) cin >> v[i];
    for (int i = 1; i <= n; i++) {  // 枚举第i个物品
        for (int j = w[i]; j <= tot; j++) {  // 枚举当前背包容量j
            dp[i][j] = dp[i-1][j];  // 默认不选择当前物品
            if (dp[i-1][j-w[i]] + v[i] > dp[i][j]) {  // 若价值更高,则选择当前物品
                dp[i][j] = dp[i-1][j-w[i]] + v[i];
                use[i][j] = 1;  // 选择当前物品
            }
        }
    }
    cout << "能获得的最大价值为: " << dp[n][tot] << '\n';
    int now = n, less = tot;
    cout << "选取物品编号: ";
    while (now) {
        if (use[now][less]) {
            cout << now << ' ';
            less -= w[now];
        }
        now--;
    }
    cout << '\n';
	return 0;
}

#if 0
输入样例：
10
5
2 2 6 5 4
6 3 5 4 6
输出：
能获得的最大价值为: 15
选取物品编号: 5 2 1
#endif
