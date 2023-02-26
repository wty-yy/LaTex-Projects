#include <iostream>
#include <string.h>
#include <algorithm>
#include <time.h>
using namespace std;
const int N = 610;
char s[3][N];  // 初始字符串
int dp[N][N][N], fa[N][N][N]; // fa={0,...,6}分别表示7个转移方向，具体方向如下面所定义
int dx[7] = {-1, -1, -1, 0, -1, 0, 0};
int dy[7] = {-1, -1, 0, -1, 0, -1, 0};
int dz[7] = {-1, 0, -1, -1, 0, 0, -1};

// dim为排除的维数，idx[0,1,2]分别为返回a-1,b-1;a-1;b-1的转移方向编号
int dim, idx[3];
// 返回a数组在排除dim维度后的i,j对应指针
int* get_prt(int a[][N][N], int i, int j) {
    if (dim == 0) return &a[0][i][j];
    else if (dim == 1) return &a[i][0][j];
    return &a[i][j][0];
}
// 返回数组a在排除dim维度后的i,j元素值
int get(int a[][N][N], int i, int j) {return *get_prt(a, i, j);}
// 设置数组a在排除dim维度后的i,j元素值
void set(int a[][N][N], int i, int j, int x) {*get_prt(a, i, j) = x;}

// 处理a,b的最短共超序列
void subsolve(char a[], char b[]) {
    int l1 = strlen(a+1), l2 = strlen(b+1);
    for (int i = 1; i <= l1; i++) {
        for (int j = 1; j <= l2; j++) {
            if (a[i] == b[j]) {
                set(dp, i, j, get(dp, i-1, j-1) + 1);
                set(fa, i, j, idx[0]);
            } else if (get(dp, i-1, j) < get(dp, i, j-1)) {
                set(dp, i, j, get(dp, i-1, j) + 1);
                set(fa, i, j, idx[1]);
            } else {
                set(dp, i, j, get(dp, i, j-1) + 1);
                set(fa, i, j, idx[2]);
            }
        }
    }
}

// 返回a,b,c的最短公共超序列
string solve(char a[], char b[], char c[]) {
    // 由于下标从1开始，所以求长度也需要指针+1
    int l1 = strlen(a+1), l2 = strlen(b+1), l3 = strlen(c+1);
    // 初始化dp数组
    for (int i = 1; i <= l1; i++) dp[i][0][0] = i, fa[i][0][0] = 4;
    for (int i = 1; i <= l2; i++) dp[0][i][0] = i, fa[0][i][0] = 5;
    for (int i = 1; i <= l3; i++) dp[0][0][i] = i, fa[0][0][i] = 6;
    // 分别求解三个二维维度上的子问题
    dim = 2, idx[0] = 1, idx[1] = 4, idx[2] = 5;
    subsolve(a, b);
    dim = 1, idx[0] = 2, idx[1] = 4, idx[2] = 6;
    subsolve(a, c);
    dim = 0, idx[0] = 3, idx[1] = 5, idx[2] = 6;
    subsolve(b, c);
    for (int i = 1; i <= l1; i++) {
        for (int j = 1; j <= l2; j++) {
            for (int k = 1; k <= l3; k++) {
                if (a[i] == b[j] && b[j] == c[k]) {
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1;
                    fa[i][j][k] = 0;
                } else if (a[i] == b[j]) {
                    dp[i][j][k] = dp[i-1][j-1][k] + 1;
                    fa[i][j][k] = 1;
                } else if (a[i] == c[k]) {
                    dp[i][j][k] = dp[i-1][j][k-1] + 1;
                    fa[i][j][k] = 2;
                } else if (b[j] == c[k]) {
                    dp[i][j][k] = dp[i][j-1][k-1] + 1;
                    fa[i][j][k] = 3;
                } else {
                    int tmp[] = {dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1]};
                    if (tmp[0] < max(tmp[1], tmp[2])) {
                        dp[i][j][k] = tmp[0] + 1;
                        fa[i][j][k] = 4;
                    } else if (tmp[1] < max(tmp[0], tmp[2])) {
                        dp[i][j][k] = tmp[1] + 1;
                        fa[i][j][k] = 5;
                    } else {
                        dp[i][j][k] = tmp[2] + 1;
                        fa[i][j][k] = 6;
                    }
                }
            }
        }
    }
    string ret;
    for (int i = l1, j = l2, k = l3; i || j || k;) {
        int f = fa[i][j][k];
        if (f == 0) ret.push_back(a[i]);
        else if (f == 1) ret.push_back(a[i]);
        else if (f == 2) ret.push_back(a[i]);
        else if (f == 3) ret.push_back(b[j]);
        else if (f == 4) ret.push_back(a[i]);
        else if (f == 5) ret.push_back(b[j]);
        else if (f == 6) ret.push_back(c[k]);
        i += dx[f], j += dy[f], k += dz[f];
    }
    reverse(ret.begin(), ret.end());  // 得到最长公共子序列lcs
    return ret;
}

int main() {
    // freopen("7-2.in", "r", stdin);
    for (int i = 0; i < 3; i++) cin >> s[i]+1; // 输入三个字符串，下标从1开始
    clock_t start = clock();
    string ans = solve(s[0], s[1], s[2]);
    clock_t end = clock();
    cout << "My Answer: " << ans << '\n';
    cout << "Length: " << ans.size() << '\n';
    cout << "Time: " << end - start << " ms";
    return 0;
}

#if 0
Input:
abed
ecaa
eacd
Output:
My Answer: ecabeacd
Length: 8

Input:
abeadeac
ecaacacc
eacebaed
Output:
My Answer: eacebaeacdeacc
Length: 14

Input:
baddbdceacaecaaabeda
dadadbeeacbeecddabce
abebcbeeaeccbeeaddad
Output:
My Answer: badadabebdcbeeaecacbeecaaddabceda
Length: 33
#endif