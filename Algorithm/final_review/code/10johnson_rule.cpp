/*
 * @Author: wty-yy
 * @Date: 2023-01-27 16:22:26
 * @LastEditTime: 2023-01-27 16:32:28
 * @Description: 验证Johnson法则是否满足传递性
 */
#include <iostream>
#include <algorithm>
using namespace std;


int n = 6, p[6], a[3], b[3];
bool check(int i, int j) {
    return min(b[i], a[j]) >= min(b[j], a[i]);
}
int main() {
    for (int i = 0; i < n; i++) p[i] = i;
    do {
        for (int i = 0; i < 3; i++) a[i] = p[i], b[i] = p[i+3];
        if (check(0, 1) && check(1, 2)) {
            if (!check(0, 2)) {
                printf("GG\n");
                printf("%d %d %d %d %d %d", a[0], b[0], a[1], b[1], a[2], b[2]);
                break;
            }
        }
        // for (int i = 0; i < n; i++) printf("%d ", p[i]);
        // putchar('\n');
    } while(next_permutation(p, p + n));
    return 0;
}
