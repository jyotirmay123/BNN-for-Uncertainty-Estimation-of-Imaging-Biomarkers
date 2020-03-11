
require(caret)
library(randomForestSRC)  
set.seed(1234)
recall <- function(matrix) {
  tp <- matrix[2, 2]
  fn <- matrix[2, 1]
  return (tp / (tp + fn))
}

cm_sanity_check <- function(matrix) {
  if(dim(matrix)[2]==1){
    matrix <- cbind(matrix, c(0,0))
    colnames(matrix)[2] <- 'TRUE'
  }
  return (matrix)
}

precision <- function(matrix) {
  tp <- matrix[2, 2]
  fp <- matrix[1, 2]
  return (tp / (tp + fp))
}

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

sample_predictor_analyser <- function(c_data, test_data, classifier_instancecvinv){
  test_data$seg_liver_scaled <- c_data
  return (predict(classifier_instancecvinv, test_data, type = "response")$class)
}

mcdata <- read.csv('~/Jyotirmay/my_thesis/projects/MC_dropout_quicknat/reports/MC_dropout_quicknat_KORA_v2/KORA/10_1572006141.7793334_concat_report_final.csv')
fbdata <- read.csv('~/Jyotirmay/my_thesis/projects/full_bayesian/reports/full_bayesian_KORA_v4/KORA/10_1572514598.527084_concat_report_final.csv')
pbdata <- read.csv('~/Jyotirmay/my_thesis/projects/probabilistic_quicknat/reports/probabilistic_quicknat_KORA_v2/KORA/10_1571996796.7963011_concat_report_final.csv')
hqdata <- read.csv('~/Jyotirmay/my_thesis/projects/hierarchical_quicknat/reports/hierarchical_quicknat_KORA_v2/KORA/10_1571905560.9377904_concat_report_final.csv')
manualdata <- read.csv('~/Jyotirmay/my_thesis/dataset_groups/whole_body_datasets/KORA/all_processed_True_concat_report_final.csv')

manualdata <- manualdata[is.element(manualdata$volume_id, mcdata$volume_id),]
manualdata <- manualdata[order(manualdata$volume_id),]
mcdata <- mcdata[order(mcdata$volume_id),]
fbdata <- fbdata[order(fbdata$volume_id),]
pbdata <- pbdata[order(pbdata$volume_id),]
hqdata <- hqdata[order(hqdata$volume_id),]

datalist = list(mcdata, fbdata, pbdata, hqdata, manualdata)

freq <- 1000
final_mean_accs <- array(0, dim=c(16, 5))
acc <- array(0, dim=c(freq, 16, 5))

for(i in 1:freq) {
  accidx = 1
  do_sample = TRUE
  for(data in datalist){
    data$diabetes_status[data$diabetes_status== 2] <- 1
    data$seg_liver_scaled = scale(data$seg_liver)
    
    data$diabetes_status <- as.factor(data$diabetes_status)
    
    if(accidx!=5){
      liv_samp <- as.matrix(data[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
      data$cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
      data$cvinv = 1/data$cv
      data$cvinv_scaled = normalize(data$cvinv)
      
      data$X0_liver_scaled = scale(data$X0_liver)
      data$X1_liver_scaled = scale(data$X1_liver)
      data$X2_liver_scaled = scale(data$X2_liver)
      data$X3_liver_scaled = scale(data$X3_liver)
      data$X4_liver_scaled = scale(data$X4_liver)
      data$X5_liver_scaled = scale(data$X5_liver)
      data$X6_liver_scaled = scale(data$X6_liver)
      data$X7_liver_scaled = scale(data$X7_liver)
      data$X8_liver_scaled = scale(data$X8_liver)
      data$X9_liver_scaled = scale(data$X9_liver)
    }
    
    if(do_sample){
      train_ids <- createDataPartition(data$diabetes_status, p = 0.5, list = FALSE, times = 1)
      test_ids <- setdiff(1:dim(data)[1],train_ids)
      do_sample = FALSE
    }
    
    print(i)
    # print(accidx)
    # print(train_ids)
    # print(test_ids)
    
    train_data <- data[train_ids, ]
    test_data  <- data[test_ids, ]
    
    classifier_base <- rfsrc(diabetes_status ~ age + sex + bmi.numeric, family='binomial', data=train_data)
    predClass <- predict(classifier_base, test_data, type = "response")$class

    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[i,1, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,2, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    classifier_vol <- rfsrc(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_vol, test_data, type = "response")$class
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[i,3, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,4, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    if(accidx==5){
      accidx = accidx + 1
      next
    }
    
    classifier_iou <- rfsrc(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled + iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_iou, test_data, type = "response")$class
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[i,5, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,6, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    classifier_cvinv <- rfsrc(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled + cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_cvinv, test_data, type = "response")$class
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[i,7, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,8, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    classifier_instanceiou <- rfsrc(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, case.wt = train_data$iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_instanceiou, test_data, type = "response")$class
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[ i,9, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,10, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    pc <- array(0, dim=c(10, nrow(test_data)))
    pc[1,] <- sample_predictor_analyser(test_data$X0_liver_scaled, test_data, classifier_instanceiou)
    pc[2,] <- sample_predictor_analyser(test_data$X1_liver_scaled, test_data, classifier_instanceiou)
    pc[3,] <- sample_predictor_analyser(test_data$X2_liver_scaled, test_data, classifier_instanceiou)
    pc[4,] <- sample_predictor_analyser(test_data$X3_liver_scaled, test_data, classifier_instanceiou)
    pc[5,] <- sample_predictor_analyser(test_data$X4_liver_scaled, test_data, classifier_instanceiou)
    pc[6,] <- sample_predictor_analyser(test_data$X5_liver_scaled, test_data, classifier_instanceiou)
    pc[7,] <- sample_predictor_analyser(test_data$X6_liver_scaled, test_data, classifier_instanceiou)
    pc[8,] <- sample_predictor_analyser(test_data$X7_liver_scaled, test_data, classifier_instanceiou)
    pc[9,] <- sample_predictor_analyser(test_data$X8_liver_scaled, test_data, classifier_instanceiou)
    pc[10,] <- sample_predictor_analyser(test_data$X9_liver_scaled, test_data, classifier_instanceiou)
    
    predClass <- colMeans(pc) > 1.5
    
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[ i,11, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,12, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    classifier_instancecvinv <- rfsrc(diabetes_status ~ age + sex + bmi.numeric + seg_liver_scaled, case.wt = train_data$cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_instancecvinv, test_data, type = "response")$class
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[ i,13, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,14, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    pc <- array(0, dim=c(10, nrow(test_data)))
    pc[1,] <- sample_predictor_analyser(test_data$X0_liver_scaled, test_data, classifier_instancecvinv)
    pc[2,] <- sample_predictor_analyser(test_data$X1_liver_scaled, test_data, classifier_instancecvinv)
    pc[3,] <- sample_predictor_analyser(test_data$X2_liver_scaled, test_data, classifier_instancecvinv)
    pc[4,] <- sample_predictor_analyser(test_data$X3_liver_scaled, test_data, classifier_instancecvinv)
    pc[5,] <- sample_predictor_analyser(test_data$X4_liver_scaled, test_data, classifier_instancecvinv)
    pc[6,] <- sample_predictor_analyser(test_data$X5_liver_scaled, test_data, classifier_instancecvinv)
    pc[7,] <- sample_predictor_analyser(test_data$X6_liver_scaled, test_data, classifier_instancecvinv)
    pc[8,] <- sample_predictor_analyser(test_data$X7_liver_scaled, test_data, classifier_instancecvinv)
    pc[9,] <- sample_predictor_analyser(test_data$X8_liver_scaled, test_data, classifier_instancecvinv)
    pc[10,] <- sample_predictor_analyser(test_data$X9_liver_scaled, test_data, classifier_instancecvinv)
    
    predClass <- colMeans(pc) > 1.5
    
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[ i,15, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,16, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    accidx = accidx + 1
  }
}

final_mean_accs[,1] = colMeans(acc[1:freq,,1])
final_mean_accs[,2] = colMeans(acc[1:freq,,2])
final_mean_accs[,3] = colMeans(acc[1:freq,,3])
final_mean_accs[,4] = colMeans(acc[1:freq,,4])
final_mean_accs[,5] = colMeans(acc[1:freq,,5])

final_mean_accs <- aperm(final_mean_accs)

write.csv(final_mean_accs, '~/Jyotirmay/my_thesis/randomforestSRC_classification_results_mean_only_same_sample_with_sample_analyser_RFSRC.csv')





