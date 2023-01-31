// 邮票问题，有n张不同面额的邮票，每张邮票数目有无限个，请问能否通过面值贴出给定邮资m？
#include <iostream>
#include <cstring>
using namespace std;

const int N = 1000;
int n, m, x[N], dp[N];
int main() {
    scanf("%d %d", &n, &m);
    for (int i = 0; i < n; i++) scanf("%d", &x[i]);
    memset(dp, 0x3f, sizeof(dp));
    dp[0] = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= m; j++)
            if (j >= x[i])
                dp[j] = min(dp[j], dp[j-x[i]]+1);
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j <= m; j++) {
    //         if (j >= x[i])
    //             dp[j] = min(dp[j], dp[j-x[i]]+1);
    //         printf("%d ", dp[j]);
    //     }
    //     printf("\n");
    // }
    return 0;
}

#if 0
4 8
1 2 4 5
#endif