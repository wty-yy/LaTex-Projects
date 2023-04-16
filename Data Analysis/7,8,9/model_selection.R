model_selection <- function(data, response, predictors) {
  # 从数据中提取出特征和真实值
  y <- data[[response]]
  X <- data[predictors]

  # 包含了所有特征的模型，用于Cp值的计算
  full_model <- lm(y ~ ., data = X)
  
  # 数据总数和特征数目
  n <- nrow(X)
  p <- length(predictors)
  
  # 初始化结果矩阵
  result <- data.frame()
  
  # 枚举特征的全部可能组合
  for (k in 1:p) {
    combos <- combn(predictors, k)
    for (i in 1:ncol(combos)) {
      # 提取当前枚举的特征
      current_predictors <- combos[, i]
      
      # 计算当前模型
      current_formula <- as.formula(paste(response, paste(current_predictors, collapse = " + "), sep = " ~ "))
      current_model <- lm(current_formula, data = data)
      
      # 计算Ra^2
      ra2 <- summary(current_model)$adj.r.squared
      
      # 计算Cp
      cp <- (sum(current_model$residuals^2) + 2 * (k + 1) * sum(full_model$residuals^2) / (n - p - 1)) / n
      
      # 计算PRESSp
      pressp <- sum((current_model$residuals / (1 - hatvalues(current_model)))^2)
      
      # 存储结果
      result <- rbind(result, data.frame(
        Predictors = paste(current_predictors, collapse = ", "),
        `Ra^2` = ra2,
        Cp = cp,
        PRESSp = pressp
      ))
    }
  }
  
  return(result)
}