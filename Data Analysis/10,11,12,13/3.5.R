data <- data.frame(
  improvement = c(7.6, 8.2, 6.8, 5.8, 6.9, 6.6, 6.3, 7.7, 6.0,
  6.7, 8.1, 9.4, 8.6, 7.8, 7.7, 8.9, 7.9, 8.3, 8.7, 7.1, 8.4,
  8.5, 9.7, 10.1, 7.8, 9.6, 9.5),
  funding = factor(c(rep("low", 9), rep("medium", 12), rep("high", 6)))
)
fit <- aov(improvement ~ funding, data = data)
# print(summary(fit))

