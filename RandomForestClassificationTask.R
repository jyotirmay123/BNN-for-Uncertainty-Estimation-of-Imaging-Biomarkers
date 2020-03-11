


library(randomForest)  

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
data$diabetes_status <- as.factor(data$diabetes_status)
data$seg_liver_scaled = scale(data$seg_liver)
liv_samp <- as.matrix(data[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
data$cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
data$cvinv = 1/data$cv

#test_data <- read.csv('~/Desktop/my_thesis/phase_2/p2_MC_dropout_quicknat_test.csv')
print('executng acc now')
rmmats <- array(0, dim=c(1005,12))

for(i in 1:1000) {
  sample <- sample.int(n = nrow(data), size = floor(.5*nrow(data)), replace = F)
  train_data <- data[sample, ]
  test_data  <- data[-sample, ]
  
  classifier_base <- randomForest(diabetes_status ~ age + sex + bmi.numeric, family='binomial', data=train_data)
  predClass <- predict(classifier_base, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass)
  print(i)
  rmmats[i,1] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  rmmats[i,2]<- 2 * ((prec * rec) / (prec + rec))
  
  classifier_vol <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, family='binomial', data=train_data)
  predClass <- predict(classifier_vol, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass)
  rmmats[i,3] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  rmmats[i,4]<- 2 * ((prec * rec) / (prec + rec))
  
  classifier_iou <- randomForest(diabetes_status ~ age + sex + bmi.numeric +seg_liver_scaled + iou_liver, family='binomial', data=train_data)
  predClass <- predict(classifier_iou, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass)
  rmmats[i,5] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  rmmats[i,6]<- 2 * ((prec * rec) / (prec + rec))
  
  classifier_cvinv <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled + cvinv, family='binomial', data=train_data)
  predClass <- predict(classifier_cvinv, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass)
  rmmats[i,7] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  rmmats[i,8]<- 2 * ((prec * rec) / (prec + rec))

  classifier_instanceiou <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, weights = train_data$iou_liver, family='binomial', data=train_data)
  predClass <- predict(classifier_instanceiou, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass)
  rmmats[i,9] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  rmmats[i,10]<- 2 * ((prec * rec) / (prec + rec))

  classifier_instancecvinv <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, weights = train_data$cvinv, family='binomial', data=train_data)
  predClass <- predict(classifier_instancecvinv, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass)
  cm
  rmmats[i,11] <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  rmmats[i,12] <- 2 * ((prec * rec) / (prec + rec))
}
rmmats[1002, ] = colMeans(rmmats[1:1000,])
#rmmats[1003,] =  apply(rmmats[1:1000,],1, sd) 

