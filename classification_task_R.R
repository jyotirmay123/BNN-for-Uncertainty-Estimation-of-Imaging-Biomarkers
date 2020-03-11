

library(dplyr)
require(caret)
#library(MLmetrics)
set.seed(1234)
recall <- function(matrix) {
  # true positive
  tp <- matrix[2, 2]# false positive
  fn <- matrix[2, 1]
  return (tp / (tp + fn))
}

precision <- function(matrix) {
  # True positive
  tp <- matrix[2, 2]
  # false positive
  fp <- matrix[1, 2]
  return (tp / (tp + fp))
}

data <- read.csv('~/Desktop/my_thesis/projects/MC_dropout_quicknat/reports/MC_dropout_quicknat_KORA_v2/KORA/10_1572006141.7793334_concat_report_final.csv')
data$diabetes_status[data$diabetes_status== 2] <- 1
liv_samp <- as.matrix(data[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
data$cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
data$cvinv = 1/data$cv

#test_data <- read.csv('~/Desktop/my_thesis/phase_2/p2_MC_dropout_quicknat_test.csv')
print('executng acc now')
acc <- array(0, dim=c(1005,12))

for(i in 1:1000) {
  sample <- sample.int(n = nrow(data), size = floor(.5*nrow(data)), replace = F)
  train_data <- data[sample, ]
  test_data  <- data[-sample, ]
  
  classifier_base <- glm(diabetes_status ~ age + sex + bmi.numeric, family='binomial', data=train_data)
  predClass <- predict(classifier_base, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass>0.5)
  print(i)
  acc[i,1] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  acc[i,2]<- 2 * ((prec * rec) / (prec + rec))
  
  classifier_vol <- glm(diabetes_status ~ age + sex + bmi.numeric + scale(seg_liver), family='binomial', data=train_data)
  predClass <- predict(classifier_vol, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass>0.5)
  acc[i,3] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  acc[i,4]<- 2 * ((prec * rec) / (prec + rec))
  
  classifier_iou <- glm(diabetes_status ~ age + sex + bmi.numeric + scale(seg_liver) + iou_liver, family='binomial', data=train_data)
  predClass <- predict(classifier_iou, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass>0.5)
  acc[i,5] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  acc[i,6]<- 2 * ((prec * rec) / (prec + rec))
  
  classifier_cvinv <- glm(diabetes_status ~ age + sex + bmi.numeric + scale(seg_liver) + cvinv, family='binomial', data=train_data)
  predClass <- predict(classifier_cvinv, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass>0.5)
  acc[i,7] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  acc[i,8]<- 2 * ((prec * rec) / (prec + rec))
  
  classifier_instanceiou <- glm(diabetes_status ~ age + sex + bmi.numeric + scale(seg_liver), weights = train_data$iou_liver, family='binomial', data=train_data)
  predClass <- predict(classifier_instanceiou, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass>0.5)
  acc[i,9] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  acc[i,10]<- 2 * ((prec * rec) / (prec + rec))
  
  # classifier_instancecvinv <- glm(diabetes_status ~ age + sex + bmi.numeric + scale(seg_liver), weights = train_data$cvinv, family='binomial', data=train_data)
  # predClass <- predict(classifier_instancecvinv, test_data, type = "response")
  # cm <- table(test_data$diabetes_status, predClass>0.5)
  # cm
  # acc[i,11] <- sum(diag(cm)) / sum(cm)
  # prec <- precision(cm)
  # rec <- recall(cm)
  # acc[i,12] <- 2 * ((prec * rec) / (prec + rec))
}
acc[1002, ] = colMeans(acc[1:1000,])
#acc[1003,] =  apply(acc[1:1000,],12, sd) 

#https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/




