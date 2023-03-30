function main
    x1 = [1, 1]';
    t = [-1, 0, 1, 2]';
    y = [2.7, 1, 0.4, 0.1]';
    A1 = A(x1, t);
    r1 = r(x1, t, y);
    delta1 = (A1' * A1) \ (A1' * r1);
    x2 = x1 + delta1;
    A2 = A(x2, t);
    r2 = r(x2, t, y);
    delta2 = (A1' * A1) \ (A2' * r2)
    x3 = delta2 + x2
end

function ret = A(x, t)
    ret = zeros(4, 2);
    for i = 1:4
        ret(i, 1) = exp(-x(2) * t(i));
        ret(i, 2) = -x(1) * t(i) * exp(-x(2) * t(i));
    end
end
function ret = r(x, t, y)
    ret = zeros(4, 1);
    for i = 1:4
        ret(i, 1) = x(1) * exp(-x(2) * t(i)) - y(i);
    end
end
