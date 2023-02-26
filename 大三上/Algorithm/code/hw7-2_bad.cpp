#include <iostream>
#include <string.h>
#include <algorithm>
using namespace std;
const int N = 1e4;
string s1, s2, s3;
int dp[N][N], fa[N][N];  // fa={0, 1, 2}分别表示从左上、上、左转移得到的，对应下面dx,dy转移坐标
int dx[3] = {-1, -1, 0}, dy[3] = {-1, 0, -1};

// 返回a与b的最短公共超序列
string solve(string a, string b) {
    int l1 = a.size(), l2 = b.size();
    for (int i = 0; i <= l1; i++)
        for (int j = 0; j <= l2; j++)
            dp[i][j] = fa[i][j] = 0;  // 初始化数组
    for (int i = 1; i <= l1; i++) {
        for (int j = 1; j <= l2; j++) {
            if (a[i-1] == b[j-1]) {  // 若两个序列字母相同
                dp[i][j] = dp[i-1][j-1] + 1;
                fa[i][j] = 0;
            }
            // 在dp[i-1][j]和dp[i][j-1]中选大的
            else if (dp[i-1][j] > dp[i][j-1]) {
                dp[i][j] = dp[i-1][j];
                fa[i][j] = 1;
            } else {
                dp[i][j] = dp[i][j-1];
                fa[i][j] = 2;
            }
        }
    }
    string lcs;
    for (int i = l1, j = l2; i && j;) {  // 逆推结果
        if (a[i-1] == b[j-1])
            lcs.push_back(a[i-1]);
        int f = fa[i][j];
        i += dx[f], j += dy[f];
    }
    reverse(lcs.begin(), lcs.end());  // 得到最长公共子序列lcs
    string ret;
    int i = 0, j = 0, k = 0;
    while (i < lcs.size()) {  // 在lcs序列中插入a和b序列即可得到a与b的最短公共超序列
        if (lcs[i] == a[j] && lcs[i] == b[k]) ret.push_back(lcs[i]), i++, j++, k++;
        if (lcs[i] != a[j]) ret.push_back(a[j]), j++;
        if (lcs[i] != b[k]) ret.push_back(b[k]), k++;
    }
    while (j < a.size()) ret.push_back(a[j++]);
    while (k < b.size()) ret.push_back(b[k++]);
    return ret;
}

// 返回a与b的最短公共超序列
int main() {
    freopen("7-2.in", "r", stdin);
    cin >> s1 >> s2 >> s3;  // 输入三个字符串
    string t = solve(s1, s2);
    cout << t << '\n';
    string ans = solve(t, s3);
    cout << "My Answer: " << ans << '\n';
    cout << "Length: " << ans.size();
    return 0;
}

#if 0
分两次计算公共超序列的错误反例
Input:
abed
ecaa
eacd
Output:
My Answer: abedcaacd
Length: 9
Answer: ecabeacd
Length: 8
#endif