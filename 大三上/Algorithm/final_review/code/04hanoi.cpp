/*
 * @Author: wty-yy
 * @Date: 2023-01-25 14:25:44
 * @LastEditTime: 2023-01-26 10:06:20
 * @Description: 求解汉诺塔移动问题.
 */
#include <iostream>
using namespace std;

// 3个塔座，从塔座a上的所有圆盘移动到塔座b上的具体方案
void hanoi3(int n, char a='A', char b='B', char c='C') {
    if (!n) return;
    hanoi3(n-1, a, c, b);
    printf("%c->%c\n", a, b);
    hanoi3(n-1, c, b, a);
}

// 4个塔座
void hanoi4(int n, int m, char a='A', char b='B', char c='C', char d='D') {
    if (n < m) {
        hanoi3(n, a, b, c);
        return;
    }
    hanoi4(n-m, m, a, d, b, c);
    hanoi3(m, a, b, c);
    hanoi4(n-m, m, d, b, a, c);
}

int main() {
    // hanoi3(3);
    hanoi4(4, 3);
    return 0;
}