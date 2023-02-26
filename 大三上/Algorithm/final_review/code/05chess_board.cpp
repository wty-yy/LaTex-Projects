/*
 * @Author: wty-yy
 * @Date: 2023-01-26 15:39:20
 * @LastEditTime: 2023-01-26 16:17:31
 * @Description: 棋盘覆盖，用四种L形骨牌，覆盖2^k x 2^k的棋盘除去其中一个点.
 */
#include <iostream>
using namespace std;

const int N = (1<<10);
int total_dominos_counter;
char board[N][N];
void fill_board(int top_x, int top_y, int out_x, int out_y, int k) {
    if (k == 0) return;
    int domino_num = ++total_dominos_counter;
    int half_size = (1 << (k-1)), domino_counter = 0, area_index;
    int mid_x = top_x + half_size, mid_y = top_y + half_size;
    // 左上棋盘
    if (out_x < mid_x && out_y < mid_y) {
        area_index = 0;
        fill_board(top_x, top_y, out_x, out_y, k-1);
    } else {
        board[mid_x-1][mid_y-1] = domino_num;
        fill_board(top_x, top_y, mid_x-1, mid_y-1, k-1);
    }
    // 右上棋盘
    if (out_x < mid_x && out_y >= mid_y) {
        area_index = 1;
        fill_board(top_x, mid_y, out_x, out_y, k-1);
    } else {
        board[mid_x-1][mid_y] = domino_num;
        fill_board(top_x, mid_y, mid_x-1, mid_y, k-1);
    }
    // 左下棋盘
    if (out_x >= mid_x && out_y < mid_y) {
        area_index = 2;
        fill_board(mid_x, top_y, out_x, out_y, k-1);
    } else {
        board[mid_x][mid_y-1] = domino_num;
        fill_board(mid_x, top_y, mid_x, mid_y-1, k-1);
    }
    // 右下棋盘
    if (out_x >= mid_x && out_y >= mid_y) {
        area_index = 3;
        fill_board(mid_x, mid_y, out_x, out_y, k-1);
    } else {
        board[mid_x][mid_y] = domino_num;
        fill_board(mid_x, mid_y, mid_x, mid_y, k-1);
    }
}

void print_board(int k);

int main() {                            
    int out_x = 2, out_y = 3, k = 2;    //  Output:
    board[out_x][out_y] = 0;            //  2  2  3  3
    fill_board(0, 0, out_x, out_y, k);  //  2  1  1  3
    print_board(k);                     //  4  1  5  0
    return 0;                           //  4  4  5  5
}

void print_board(int k) {
    int n = 1 << k;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%-3d", board[i][j]);
        }
        putchar('\n');
    }
}