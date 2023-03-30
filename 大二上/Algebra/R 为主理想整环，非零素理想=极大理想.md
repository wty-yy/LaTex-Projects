$R$ 为主理想整环，$P$ 为 $R$ 中的理想，证明：$P$ 为非零素理想 $\iff$ $P$ 为极大理想。

**证明**： “$\Leftarrow$”： 显然。

“$\Rightarrow$”： 令 $P$ 为 $R$ 中的非零素理想，则 $\exists\ a\in R,\ a\neq 0$，使得 $P = (a)$，设存在 $R$ 的一个理想 $I$，使得 $P\subset I$，则 $\exists\ b\in R$，使得 $I = (b)$。下证 $I = R$ 或者 $I = P$。

由于 $(a)\subset (b)$，则 $\exists\ r\in R$，使得 $a = rb\Rightarrow r\in (a)或者 b\in(a)$。

1. 若 $b\in(a)$，则 $(b)\subset (a)$，故 $(a) = (b)\Rightarrow I = P$。
2. 若 $r\in (a)$，则 $\exists\ t\in R$，使得 $r = ta$，则 $a = tab\Rightarrow (tb-1)a = 0$，由于 $R$ 为整环，所以 $tb-1=0或a=0$，则 $tb = 1\in (b)$，则 $(b) = R$，则 $I = R$。

综上，$P$ 为 $R$ 中的极大理想。