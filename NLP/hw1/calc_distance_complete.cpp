#include <iostream>
#include <cstring>
using namespace std;
const int N = 1e4 + 10;  // 字符串最大长度
char s1[N], s2[N];
int dp[N][N], opt[N][N];  // dp为动态规划数组, opt记录每次操作值
// 0: none, 1: delete, 2: add, 3: change
string optName[] = {"none", "del", "add", "chg"}, nowString;
int delta = 0;
void dfs(int i, int j, int o) {
    if (opt[i][j] == 0) dfs(i-1, j-1, 0);
    else if (opt[i][j] == 1) dfs(i-1, j, 1);
    else if (opt[i][j] == 2) dfs(i, j-1, 2);
    else if (opt[i][j] == 3) dfs(i-1, j-1, 3);
    if (o != 0) {
        if (o == 1) nowString.erase(i+delta, 1), delta--;
        else if (o == 2) nowString.insert(i+delta, 1, s2[j]), delta++;
        else nowString[i+delta] = s2[j];
        cout << optName[o] << ": " << nowString << '\n';
    }
}
signed main() {
    memset(dp, 127, sizeof(dp));  // 初始化dp数组为极大值
    memset(opt, -1, sizeof(opt));  // 初始化opt数组为-1
    cin >> s1 >> s2;
    int len1 = strlen(s1), len2 = strlen(s2);
    for (int i = 0; i <= max(len1, len2); i++) dp[0][i] = dp[i][0] = i;
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            if (s1[i-1] == s2[j-1]) {
                opt[i][j] = 0;
                dp[i][j] = dp[i-1][j-1];
                continue;
            }
            int tmp[] = {dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}, mn = 1e9, mnId;  // mn记录当前最小值，mnId记录从哪个位置转移得到的
            for (int k = 0; k < 3; k++) {
                if (tmp[k] < mn) {
                    mn = tmp[k];
                    mnId = k;
                }
            }
            dp[i][j] = mn + 1;
            opt[i][j] = mnId + 1;
        }
    }
    printf("minimal step: %d\n", dp[len1][len2]);
    //---------- 以上部分为动态规划部分 ----------
    nowString = string(s1);
    cout << "s1:  " << nowString << '\n';
    dfs(len1, len2, 0);
    system("pause");
    return 0;
}

#if 0
Input:
aadfagha
abcdefgh

Output:
minimal step: 5
s1:  aadfagha
chg: abdfagha
add: abcdfagha
add: abcdefagha
del: abcdefgha
del: abcdefgh
#endif