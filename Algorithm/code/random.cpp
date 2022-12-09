#include <iostream>
#include <time.h>
using namespace std;

int main() {
    srand(time(NULL));
    freopen("in.in", "w", stdout);
    int n = 20;
    cout << n << '\n';
    for (int i = 0; i < n; i++) {
        printf("%d ", rand() % 100);
    }
    cout << '\n';
    return 0;
}