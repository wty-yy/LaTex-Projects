/*
 * @Author: wty-yy
 * @Date: 2023-01-29 09:50:28
 * @LastEditTime: 2023-01-29 09:59:56
 * @Description: n皇后问题
 */
#include <iostream>
using namespace std;

const int N = 100;
int n=8, sum;
bool column[N], diag1[N], diag2[N];
void n_queens(int i) {
    if (i >= n) sum++;
    else {
        for (int j = 0; j < n; j++) {
            if (diag1[i+j] || diag2[i-j+n] || column[j])
                continue;
            diag1[i+j] = diag2[i-j+n] = column[j] = 1;
            n_queens(i+1);
            diag1[i+j] = diag2[i-j+n] = column[j] = 0;
        }
    }
}

int main() {
    n_queens(0);
    printf("%d\n", sum);
    return 0;
}