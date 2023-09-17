setwd("/home/wty/Documents/LaTex-Projects/Data Analysis/3,4,5,6")
data <- read.table(file = "data.csv", sep = ",")
X <- data[1:10, ]
Y <- data[11:20, ]
print(t.test(X, Y, var.equal = FALSE, paired = FALSE, mu = 0))
# var.equal为FALSE表示假设两个总体方差不相等，paired为FALSE表示假设两个样本是独立的，mu为0表示检验两个总体均值是否相等