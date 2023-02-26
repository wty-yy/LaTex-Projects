#include <iostream>
#include <cstring>
using namespace std;
const int N = 1000;
int n, m, d, a[N], b[N], dp[N][N], ans;
int main() {
    memset(dp, 128, sizeof(dp));  // 初始化dp为负最大值
    cin >> n >> d >> m;
    for (int i = 0; i < n; i++) cin >> a[i] >> b[i];
    dp[0][0] = a[0];
    for (int j = 1; j <= m; j++) {  // j为当前跳跃次数
        for (int i = 1; i < n; i++) {  // i为当前位于树的编号
            for (int k = i-1; k >= 0 && b[i] - b[k] <= d; k--) {  // 枚举从第k棵树上跳过去
                dp[i][j] = max(dp[k][j-1] + a[i], dp[i][j]);
                ans = max(ans, dp[i][j]);
            }
        }
    }
    cout << ans << '\n';
	return 0;
}
#if 0
输入：
6 8 2
3 0
6 5
1 6
2 7
7 10
4 11
输出：
16
#endif
