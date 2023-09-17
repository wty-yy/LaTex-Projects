data <- data.frame(
  A = rep(c("低剂量", "中剂量", "高剂量"), each = 12),
  B = rep(rep(c("低剂量", "中剂量", "高剂量"), each = 4), 3),
  value = c(2.4,2.7,2.3,2.5,4.6,4.2,4.9,4.7,4.8,4.5,4.4,4.6,
            5.8,5.2,5.5,5.3,8.9,9.1,8.7,9,9.1,9.3,8.7,9.4,
            6.1,5.7,5.9,6.2,9.9,10.5,10.6,10.1,13.5,13.0,
            13.3,13.2)
)

library(dplyr)
data_summary <- data %>%
  group_by(A,B) %>%
  summarise(mean_value = mean(value))

library(ggplot2)
myplot <- ggplot(data_summary,aes(x=B,y=mean_value,color=A)) +
  geom_point() +
  geom_line() +
  facet_wrap(~A) +
  labs(title="各组和水平上的样本均值",x="成分B",y="均值")
print(myplot)
