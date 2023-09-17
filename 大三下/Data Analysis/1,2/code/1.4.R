setwd("E:/Coding/R/习题一")
data <- read.table("exercise1_4.txt", fileEncoding = "GB2312")
x1 <- data[, 2]
x2 <- data[, 3]
cat("x1 =", x1, "\nx2 =", x2, "\n")

# 计算均值
mean_x1 <- mean(x1)
mean_x2 <- mean(x2)

# 计算方差
var_x1 <- var(x1)
var_x2 <- var(x2)

# 计算标准差
sd_x1 <- sd(x1)
sd_x2 <- sd(x2)

# 计算变异系数
cv_x1 <- sd_x1 / mean_x1 * 100
cv_x2 <- sd_x2 / mean_x2 * 100

# 计算偏度
skewness_x1 <- moments::skewness(x1)
skewness_x2 <- moments::skewness(x2)

# 计算峰度
kurtosis_x1 <- moments::kurtosis(x1)
kurtosis_x2 <- moments::kurtosis(x2)

# 输出结果
cat("x1: 均值 = ", mean_x1, ", 方差 = ", var_x1, ", 标准差 = ", sd_x1,
    ", 变异系数 = ", cv_x1, ", 偏度 = ", skewness_x1, ", 峰度 = ", kurtosis_x1,
    "\n", sep = "")
cat("x2: 均值 = ", mean_x2, ", 方差 = ", var_x2, ", 标准差 = ", sd_x2,
    ", 变异系数 = ", cv_x2, ", 偏度 = ", skewness_x2, ", 峰度 = ", kurtosis_x2,
    "\n", sep = "")


# 计算上下四分位数及中位数
q1_x1 <- quantile(x1, probs = 0.25)
median_x1 <- quantile(x1, probs = 0.5)
q3_x1 <- quantile(x1, probs = 0.75)

q1_x2 <- quantile(x2, probs = 0.25)
median_x2 <- quantile(x2, probs = 0.5)
q3_x2 <- quantile(x2, probs = 0.75)

# 计算四分位极差
iqr_x1 <- q3_x1 - q1_x1
iqr_x2 <- q3_x2 - q1_x2

cat("x1: 上四分位数 = ", q1_x1, ", 中位数 = ", median_x1, ", 下四分位数 = ", q3_x1,
    "\n", sep = "")
cat("x2: 上四分位数 = ", q1_x2, ", 中位数 = ", median_x2, ", 下四分位数 = ", q3_x2,
    "\n", sep = "")

png("x1_histogram.png", width = 800, height = 600, res = 96)
hist(x1, main = "x1 直方图", xlab = "x1", ylab = "频数", col = "lightblue")
dev.off()
png("x2_histogram.png", width = 800, height = 600, res = 96)
hist(x2, main = "x2 直方图", xlab = "x2", ylab = "频数", col = "#f77e92")
dev.off()

# 计算两组样本的ECDF函数
ecdf_x1 <- ecdf(x1)
ecdf_x2 <- ecdf(x2)

# 绘制ECDF图
png("ecdf.png", width = 600, height = 800, res = 96)
plot(ecdf_x1, main = "经验分布函数图", xlab = "x", ylab = "F(x)", col = "blue")
lines(ecdf_x2, col = "red")
legend("right", legend = c("x1", "x2"), col = c("blue", "red"), lty = 1)
dev.off()

# Pearson相关系数和Spearman相关系数
cor_pearson <- cor(x1, x2, method = "pearson")
cor_spearman <- cor(x1, x2, method = "spearman")

cat("Pearson相关系数 = ", cor_pearson, ", Spearman相关系数 = ", cor_spearman,
    "\n", sep = "")