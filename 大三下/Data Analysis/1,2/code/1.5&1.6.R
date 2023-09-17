setwd("E:/Coding/R/习题一")
data <- read.table("exercise1_5.txt")

# 总体均值向量
data_mean <- colMeans(data)
print(data_mean)
# 总体协方差矩阵
data_cov <- cov(data)
print(data_cov)

# 中位数向量
data_median <- apply(data, 2, median)
cat("中位数向量\n")
print(data_median)

# Pearson相关矩阵
R <- cor(data, method = "pearson")
cat("Pearson相关矩阵\n")
print(R)
# Spearman相关矩阵
Q <- cor(data, method = "spearman")
cat("Spearman相关矩阵\n")
print(Q)

# 计算Pearson两两列做相关性分析
pearson_values <- matrix(nrow = ncol(data), ncol = ncol(data))
spearman_values <- matrix(nrow = ncol(data), ncol = ncol(data))
for (j in 1:ncol(data)) {
    for (k in 1:ncol(data)) {
        pearson_test <- cor.test(data[, j], data[, k])
        pearson_values[j, k] <- pearson_test$p.value
        spearman_test <- cor.test(data[, j], data[, k], method = "spearman")
        spearman_values[j, k] <- spearman_test$p.value
    }
}
cat("Pearson检验\n")
print(pearson_values)
cat("Spearman检验\n")
print(spearman_values)