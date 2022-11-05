#include <iostream>
#include <cstring>
using namespace std;
const int N = 1e4 + 10;  // 字符串最大长度
char s1[N], s2[N];
int dp[N][N];  // dp为动态规划数组
signed main() {
    memset(dp, 127, sizeof(dp));  // 初始化dp数组为极大值
    cin >> s1 >> s2;  // 字符串读入
    int len1 = strlen(s1), len2 = strlen(s2);
    for (int i = 0; i <= max(len1, len2); i++) dp[0][i] = dp[i][0] = i;  // 初始化边界条件
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
                continue;
            }
            dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1;
        }
    }
    printf("%d\n", dp[len1][len2]);
    system("pause");
    return 0;
}