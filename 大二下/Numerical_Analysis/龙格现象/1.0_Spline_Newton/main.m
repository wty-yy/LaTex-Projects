function main
    format short
    a = -1;
    b = 1;
    x = (a:0.01:b)';
    plot(x, f(x), 'r', 'linewidth', 1); % f(x)图像
    hold on
    % Newton插值法求N5,N10
    xx = (a:2/5:b)';
    y = Newton(xx, f(xx));
    class(y)
    disp("N5 = ")
    disp(y) % 输出N5插值结果
    plot(x, subs(y, x), 'color', '#43A047', 'linewidth', 1); % N5图像
    hold on
    xx = (a:2/10:b)';
    y = Newton(xx, f(xx));
    disp("N10 = ")
    disp(y) % 输出N10插值结果
    plot(x, subs(y, x), 'linewidth', 1); % N10图像
    hold on
    % 三次样条插值
    xx = (a:2/10:b)';
    y = mySpline(xx, f(xx), fff(a), fff(b));
    disp("S10 = ")
    disp(y)
    for i = 1 : 10
        l = xx(i);
        r = l + 2/10;
        x = l:0.01:r;
        plot(x, subs(y(i), x), 'b', 'linewidth', 1) % S10图像
        hold on
    end
    legend(["f(x)" "N5" "N10" "S10"]);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 16);
end

function y = f(x)
    y = 1 ./ (1 + 25 * x .* x);
end
function y = fff(x)
    y = 50 * (75*x.*x-1) ./ (1+25*x.*x)^3;
end

% Newton插值法,a为插值点列向量,b为插值点对应的函数值列向量
function N = Newton(a, b)
    [n, ~] = size(a);
    diff = zeros(n, n); % 初始化差商表
    diff(1, :) = b';
    syms x % 定义参数x
    now = 1; % 对应上文中的 pi(x)
    N = diff(1, 1);
    % 递推计算差商表，并同时计算出插值多项式N(x)
    for i = 2 : n
        for j = i : n
            diff(i, j) = (diff(i-1, j)-diff(i-1, j-1)) / (a(j) - a(j-i+1));
        end
        now = now * (x-a(i-1));
        N = N + diff(i, i) * now;
    end
    N = expand(N); % 展开插值多项式
    N = vpa(N, 5); % 将分式系数转化为小数系数
end

% 三次样条插值法
% a为传入的插值点构成的向量,b为插值点对应函数值构成的向量,M1,Mn为边界条件
function S = mySpline(a, b, M1, Mn)
    [n, ~] = size(a);
    % 向量定义
    h = a(2:n) - a(1:n-1); % 求解向量h
    mu = h(1:n-2) ./ (h(1:n-2) + h(2:n-1)); % 求解向量mu
    la = 1 - mu; % 求解向量lambda
    h1 = h(1:n-2); % 定义中间变量h1
    h2 = h(2:n-1); % 定义中间变量h2
    % 求解向量d
    d = 6*((b(3:n)-b(2:n-1))./h2-(b(2:n-1)-b(1:n-2))./h1)./(h1+h2);

    % 生成系数矩阵A，修改d向量，利用追赶法求解M
    A = zeros(n-2, n-2); % 初始化系数矩阵A
    for i = 1 : n-2 % 循环构造系数矩阵A
        if (i > 1)
            A(i, i-1) = mu(i); % 对角线下方
        end
        A(i, i) = 2; % 对角线
        if (i < n-2)
            A(i, i+1) = la(i); % 对角线上方
        end
    end
    d(1) = d(1) - mu(1) * M1; % 修改向量d
    d(n-2) = d(n-2) - la(n-2) * Mn; % 修改向量d
    M = [M1; chase(A, d)'; Mn]; % 利用追赶法求解方程组,并解出向量M

    % 求解三次样条插值函数S(x)
    syms x % 定义参数x
    % 根据上述定义,计算出以下6个中间变量
    x1 = a(2:n) - x;
    x2 = x - a(1:n-1);
    m1 = M(1:n-1);
    m2 = M(2:n);
    y1 = b(1:n-1);
    y2 = b(2:n);
    % 求解三次样条插值函数
    S = (x1.^3).*m1./(6*h)+(x2.^3).*m2./(6*h)+(y1-h.^2.*m1/6).*x1./h+(y2-h.^2.*m2/6).*x2./h;
    S = expand(S); % 展开插值多项式
    S = vpa(S, 5); % 将分式系数转化为小数系数
end

% 求解三对角方程组的追赶法
function x = chase(A, d) % A为系数矩阵,d为方程组右侧列向量
    [n, ~] = size(A);
    % 预分配内存,提高运算速度
    u = zeros(n, 1);
    l = zeros(n, 1);
    y = zeros(n, 1);
    % 参考课本第36面代码
    u(1) = A(1, 1);
    y(1) = d(1);
    for i = 2 : n
        l(i) = A(i, i-1) / u(i-1);
        u(i) = A(i, i) - l(i) * A(i-1, i);
        y(i) = d(i) - l(i) * y(i-1);
    end
    x(n) = y(n) / u(n);
    for i = n-1 : -1 : 1
        x(i) = (y(i) - A(i, i+1) * x(i+1)) / u(i);
    end
end