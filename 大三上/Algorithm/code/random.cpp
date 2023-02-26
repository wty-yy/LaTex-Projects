#include <iostream>
#include <time.h>
#include <algorithm>
using namespace std;

// 此代码用于生成随机测试数据，将main后面的序号去掉就可以生成该题对应的数据

int main() {  // 7-1随机生成数据
    srand(time(NULL));
    freopen("7-1.in", "w", stdout);
    int n = 30;
    cout << n << '\n';
    for (int i = 0; i < n; i++) {
        printf("%d ", rand() % 100);
    }
    cout << '\n';
    return 0;
}

int main7_2() {  // 7-2随机生成数据
    srand(time(NULL));
    freopen("7-2.in", "w", stdout);
    int n = 300;  // 每个序列的长度均为n
    string s[3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < n; j++)
            s[i].push_back(rand() % 10 + 'a');
        cout << s[i] << '\n';
    }
    // freopen("7-2-num.in", "w", stdout);
    // printf("%d\n", n);
    // for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%d ", s[i][j] - 'a');
    //     }
    //     putchar('\n');
    // }
    return 0;
}

int x[1000000], y[1000000];
int main7_3() {  // 7-3随机生成数据
    srand(time(NULL));
    freopen("7-3.in", "w", stdout);
    int n = 10;  // 总点数目，生成点范围在[0,...,n), [0,...,n)中间，保证两两之间横纵坐标不同
    cout << n << '\n';
    for (int i = 0; i < n; i++) x[i] = y[i] = i;
    random_shuffle(x, x + n);
    random_shuffle(y, y + n);
    for (int i = 0; i < n; i++) cout << x[i] << ' ' << y[i] << '\n';
    return 0;
}