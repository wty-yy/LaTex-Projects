#include <cmath>
#include <ctime>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <algorithm>
#define reset(A, a) std::memset(A, (a), sizeof(A))
#define rep(i, n) for (int i = 0; i < n; i++)
#define _rep(i, n) for (int i = 1; i <= n; i++)

const int maxn = 10 + 10;   // 城市数上界
const int maxm = 10 + 10;   // 蚂蚁数上界
const int INIT_TAU = 10;    // 信息素初值
const int maxt = 100;         // 迭代重复次数
double ALPHA = 1;     // 信息素比重
double BETA = 5;      // 启发信息(城市距离)比重
const double EPS = 1e-8;    // 浮点数精度
double RHO = 0.5;     // 信息素残留率
double Q = 10;        // 信息素更新时相关常数
double BEST_DIS = 2.69067064;
int n, m = 10;
std::pair<double, double> citys[maxn];
double dis[maxn][maxn];     // 邻接表存图
void read_data() {
    freopen("citys.txt", "r", stdin);
    scanf("%d", &n);
    rep(i, n) scanf("%lf %lf", &citys[i].first, &citys[i].second);
    fclose(stdin);
    rep(i, n) rep(j, n) {
        double d1 = citys[i].first - citys[j].first, d2 = citys[i].second - citys[j].second;
        dis[i][j] = std::sqrt(d1 * d1 + d2 * d2);
    }
}
void show_path(int *path) { rep(i, n+1) printf("%d ", path[i]); putchar('\n'); }
struct BestState {  // 记录最优路径
    double dis; int path[maxn];
    BestState() { init(); }
    void init() { dis = 1e9; }
    void update(double d, int *p) {
        if (d > dis) return;
        dis = d;
        rep(i, n+1) path[i] = p[i];
    }
    void print() {
        printf("Minimal distance: %.8lf\nBest path: ", dis);
        show_path(path);
    }
}best;
int pos[maxm];          // 蚂蚁位置
int path[maxm][maxn];   // 蚂蚁经过的城市
double pathdis[maxm];   // 蚂蚁当前路线长度
bool vis[maxm][maxn];   // 蚂蚁访问过的城市
double tau[maxn][maxn]; // 每条边上的信息素
void debug() {
    rep(i, n) printf("pathdis[%d]: %lf\n", i, pathdis[i]);
    best.print();
    printf("tau:\n");
    rep(i, n) { rep(j, n) printf("%5.2lf ", tau[i][j]); putchar('\n'); }
    freopen("best_path.txt", "w", stdout);
    show_path(best.path);
    fclose(stdout);
    freopen("/dev/tty", "w", stdout);  // 转到控制台输出
}
void update() {  // 进行一轮信息素更新
    reset(pathdis, 0); reset(vis, 0);
    rep(i, m) {  // 初始化蚂蚁信息
        pos[i] = std::rand() % n;
        path[i][0] = pos[i];
        vis[i][pos[i]] = 1;
    }
    double prob[maxn];  // 临时数组，计算转移概率
    for (int t = 1; t <= n; t++) {
        rep(i, m) {
            double sum = 0; int next = -1; reset(prob, 0);
            if (t == n) next = path[i][0];
            else {
                rep(j, n) if (!vis[i][j]) {  // 计算转移概率
                    prob[j] = std::pow(tau[pos[i]][j], ALPHA) * std::pow(1.0/dis[pos[i]][j], BETA);
                    sum += prob[j];
                }
                rep(j, n) prob[j] /= sum;
                double r = (double)std::rand() / RAND_MAX;  // 根据随机数求转移城市
                rep(j, n) if (!vis[i][j]) {
                    if (r <= prob[j] + EPS) { next = j; break; }
                    else r -= prob[j];
                }
            }
            assert(next != -1);
            pathdis[i] += dis[pos[i]][next];
            pos[i] = next; path[i][t] = next; vis[i][next] = 1;  // 移动
        }
    }
    rep(i, n) rep(j, n) tau[i][j] *= 1 - RHO;
    rep(i, m) {
        best.update(pathdis[i], path[i]);  // 更新最优解
        rep(j, n) {
            int u = path[i][j], v = path[i][j+1];
            tau[u][v] += Q / pathdis[i];  // 更新每条边上的信息素
            tau[v][u] += Q / pathdis[i];  // 更新每条边上的信息素
        }
    }
    // debug();
}
int solve() {
    int T;
    best.init();
    rep(i, n) rep(j, n) tau[i][j] = INIT_TAU;
    for (T = 1; T <= maxt; T++) {
        update();
        if (std::abs(best.dis - BEST_DIS) < EPS) break;
    }
    return T;
}
void make_test() {
    freopen("test_log.txt", "w", stdout);
    int test = 1e3;
    printf("ALPHA=7, Test BETA:\n");
    for (BETA = 1; BETA <= 10; BETA++) {
        double avg = 0;
        for (int i = 0; i < test; i++) avg += solve();
        avg /= test;
        printf("%3.0lf %5.2lf\n", BETA, avg);
    }
    BETA = 5;
    printf("BETA=5, Test ALPHA:\n");
    for (ALPHA = 0; ALPHA <= 5; ALPHA++) {
        double avg = 0;
        for (int i = 0; i < test; i++) avg += solve();
        avg /= test;
        printf("%3.0lf %5.2lf\n", ALPHA, avg);
    }
    ALPHA = 1;
    printf("ALPHA=1, BETA=5, Test rho:\n");
    for (RHO = 0; RHO <= 0.9; RHO += 0.1) {
        double avg = 0;
        for (int i = 0; i < test; i++) avg += solve();
        avg /= test;
        printf("%3.2lf %5.2lf\n", RHO, avg);
    }
    RHO = 0.5;
    printf("ALPHA=1, BETA=5, RHO=0.5, Test Q:\n");
    for (Q = 0; Q <= 100; Q += 10) {
        double avg = 0;
        for (int i = 0; i < test; i++) avg += solve();
        avg /= test;
        printf("%3.0lf %5.2lf\n", Q, avg);
    }
}
int main() {
    // srand(2023);  // 随机种子
    srand(time(NULL));  // 随机种子
    read_data();
    make_test();
    return 0;
}

