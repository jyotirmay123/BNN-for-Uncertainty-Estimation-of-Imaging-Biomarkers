

require(caret)
library(randomForest)  
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

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

manualdata <- read.csv('~/Jyotirmay/my_thesis/dataset_groups/whole_body_datasets/KORA/all_processed_True_concat_report_final.csv')
mcdata <- read.csv('~/Jyotirmay/my_thesis/projects/MC_dropout_quicknat/reports/MC_dropout_quicknat_KORA_v2/KORA/10_1572006141.7793334_concat_report_final.csv')
fbdata <- read.csv('~/Jyotirmay/my_thesis/projects/full_bayesian/reports/full_bayesian_KORA_v4/KORA/10_1572514598.527084_concat_report_final.csv')
pbdata <- read.csv('~/Jyotirmay/my_thesis/projects/probabilistic_quicknat/reports/probabilistic_quicknat_KORA_v2/KORA/10_1571996796.7963011_concat_report_final.csv')
hqdata <- read.csv('~/Jyotirmay/my_thesis/projects/hierarchical_quicknat/reports/hierarchical_quicknat_KORA_v2/KORA/10_1571905560.9377904_concat_report_final.csv')

datalist = list(manualdata, mcdata, fbdata, pbdata, hqdata)
final_mean_accs <- array(0, dim=c(5,12))
accidx = 1
for(data in datalist){
  
  data$diabetes_status[data$diabetes_status== 2] <- 1
  data$diabetes_status <- as.factor(data$diabetes_status)
  data$seg_liver_scaled = scale(data$seg_liver)
  
  if(accidx!=1){
    liv_samp <- as.matrix(data[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
    data$cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
    data$cvinv = 1/data$cv
    data$cvinv_scaled = normalize(data$cvinv)
  }
  
  print('executng acc now')
  acc <- array(0, dim=c(105,12))
  
  for(i in 1:100) {
    print(i)
    sample <- sample.int(n = nrow(data), size = floor(.5*nrow(data)), replace = F)
    
    train_ids <- createDataPartition(data$diabetes_status, p = 0.5, list = FALSE, times = 1)
    test_ids <- setdiff(1:dim(data)[1],train_ids)
    
    train_data <- data[train_ids, ]
    test_data  <- data[test_ids, ]
    
    classifier_base <- randomForest(diabetes_status ~ age + sex + bmi.numeric, family='binomial', data=train_data)
    predClass <- predict(classifier_base, test_data, type = "response")
    print(predClass)
    cm <- table(test_data$diabetes_status, predClass)
    print(cm)
    acc[i,1] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,2]<- 2 * ((prec * rec) / (prec + rec))
    
    classifier_vol <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_vol, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    acc[i,3] <- sum(diag(cm)) / sum(cm)
    print(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,4]<- 2 * ((prec * rec) / (prec + rec))
    
    if(accidx==1) next
    
    classifier_iou <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled + iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_iou, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    acc[i,5] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,6]<- 2 * ((prec * rec) / (prec + rec))
    
    classifier_cvinv <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled + cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_cvinv, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    acc[i,7] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,8]<- 2 * ((prec * rec) / (prec + rec))
    
    classifier_instanceiou <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, weights = train_data$iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_instanceiou, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    acc[i,9] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,10]<- 2 * ((prec * rec) / (prec + rec))
    
    classifier_instancecvinv <- randomForest(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, weights = train_data$cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_instancecvinv, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    acc[i,11] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,12] <- 2 * ((prec * rec) / (prec + rec))
  }
  final_mean_accs[accidx, ] = colMeans(acc[1:100,])
  accidx = accidx + 1
  #acc[1003,] =  apply(acc[1:1000,],12, sd) 
}

write.csv(final_mean_accs, '~/Jyotirmay/my_thesis/randomforest_classification_results_mean_only_.csv')

#https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/




