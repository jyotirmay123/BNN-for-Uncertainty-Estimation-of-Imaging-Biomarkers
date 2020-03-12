
require(caret)
require(randomForest)
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

sample_predictor_analyser <- function(c_data, test_data, classifier_instancecvinv){
  test_data$seg_liver_scaled <- c_data
  return (predict(classifier_instancecvinv, test_data, type = "response"))
}
precision <- function(matrix) {
  
  tp <- matrix[2, 2]
  fp <- matrix[1, 2]
  return (tp / (tp + fp))
}

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

remove_duplicates <- function(data){
  # Removing all entries of a duplicate volumes.
  duplicates <- data[duplicated(data$volume_id), ]$volume_id
  if(length(duplicates)){
    data <- data[!is.element(data$volume_id, duplicates),]
  }
  return (data)
}

remove_nans <- function(data){
  return (data[complete.cases(data$seg_liver),])
}

get_common_vols <- function(datasets){
  start_flag <- TRUE
  for(data in datasets){
    data <- remove_duplicates(data)
    data <- remove_nans(data)
    if(start_flag){
      start_flag <- FALSE
      common_vols <- data$volume_id
    } else {
      common_vols = intersect(common_vols, data$volume_id)  
    }
  }
  return (common_vols)
}

truncate_with_vols <- function(data, vols_list){
  data <- data[is.element(data$volume_id, vols_list),]
  return (data)
}

ukb_preprocess <- function(data, common_vols, is_manual=FALSE){
  
  if(!is_manual){
    data <- truncate_with_vols(data, common_vols)
    data <- data[order(data$volume_id),]
  }
  
  names(data)[names(data) == "bmi_numeric"] <- "bmi.numeric"
  data$diabetes_status[data$diabetes_status== 2] <- 1
  data$seg_liver_scaled = scale(data$seg_liver)
  data$diabetes_status_l <- as.logical(data$diabetes_status)
  data$diabetes_status <- as.factor(data$diabetes_status)
  
  if(!is_manual){
    liv_samp <- as.matrix(data[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
    data$cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
    # Still few CV with NANs. removing them here. Recheck Coomon_volumes across the sets.
    data <- data[complete.cases(data$cv),]
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
  return (data)
}

get_appropriate_ukb_dataset <- function(dataset_id){
  if(dataset_id==1){
    print('Fetching MC Drop ukb data')
    return (mcukbdata)
  } else if(dataset_id==2){
    print('Fetching Bayesian data')
    return (fbukbdata)
  }else if(dataset_id==3){
    print('Fetching Probabilistic ukb data')
    return (pbukbdata)
  }else if(dataset_id==4){
    print('Fetching Hierarchical ukb data')
    return (hqukbdata)
  }else if(dataset_id==5){
    print('Fetching Manual ukb data')
    return (manualukbdata)
  }
}


mcukbdata <- read.csv('~/Jyotirmay/my_thesis/projects/MC_dropout_quicknat/reports/MC_dropout_quicknat_UKB_v2/UKB/0_0.0_concat_report_final_all.csv')
fbukbdata <- read.csv('~/Jyotirmay/my_thesis/projects/full_bayesian/reports/full_bayesian_UKB_v4/UKB/10_1574676555.7948809_concat_report_final_pp_final.csv')
pbukbdata <- read.csv('~/Jyotirmay/my_thesis/projects/probabilistic_quicknat/reports/probabilistic_quicknat_UKB_v2/UKB/10_1573834823.1121247_concat_report_final.csv')
hqukbdata <- read.csv('~/Jyotirmay/my_thesis/projects/hierarchical_quicknat/reports/hierarchical_quicknat_UKB_v2/UKB/10_1574308007.2486243_concat_report_final.csv')
manualukbdata <- read.csv('~/Jyotirmay/my_thesis/dataset_groups/whole_body_datasets/UKB/all_processed_True_concat_report_final.csv')

#666 diabetic

ukbdatalist = list(mcukbdata, fbukbdata, pbukbdata, hqukbdata)
common_vols <- get_common_vols(ukbdatalist)
mcukbdata <- ukb_preprocess(mcukbdata, common_vols)
fbukbdata <- ukb_preprocess(fbukbdata, common_vols)
pbukbdata <- ukb_preprocess(pbukbdata, common_vols)
hqukbdata <- ukb_preprocess(hqukbdata, common_vols)

ukbprocessedlist = list(mcukbdata, fbukbdata, pbukbdata, hqukbdata)
common_vols <- get_common_vols(ukbprocessedlist)

# Redo because of cvinv NaN cases!!!
mcukbdata <- truncate_with_vols(mcukbdata, common_vols)
fbukbdata <- truncate_with_vols(fbukbdata, common_vols)
pbukbdata <- truncate_with_vols(pbukbdata, common_vols)
hqukbdata <- truncate_with_vols(hqukbdata, common_vols)

manualukbdata <- ukb_preprocess(manualukbdata, common_vols = FALSE, is_manual = TRUE)

# Creating dataset with equal # of diabetic and non-diabetic volumes.

tmp <- mcukbdata[,c("diabetes_status_l","volume_id", "age","sex")]
match.it <- matchit(diabetes_status_l ~ age + sex, data = tmp, method="nearest", ratio=1)
matchMatrix = match.it$match.matrix
matched_volumes = tmp[matchMatrix,]$volume_id

get_optimal_dataset <- function(data, matched_volumes){
  data_non_diabetic <- data[is.element(data$volume_id, matched_volumes),]
  data_diabetic <- data[data$diabetes_status == 1,]
  data <- rbind(data_non_diabetic, data_diabetic)
  data <- data[order(data$volume_id),]
  return (data)
}
mcukbdata <- get_optimal_dataset(mcukbdata, matched_volumes)
fbukbdata <- get_optimal_dataset(fbukbdata, matched_volumes)
pbukbdata <- get_optimal_dataset(pbukbdata, matched_volumes)
hqukbdata <- get_optimal_dataset(hqukbdata, matched_volumes)

datalist = list(mcukbdata, fbukbdata, pbukbdata, hqukbdata, manualukbdata)

freq <- 100
final_mean_accs <- array(0, dim=c(16, 5))
acc <- array(0, dim=c(freq, 16, 5))

for(i in 1:freq) {
  accidx = 1
  do_sample = TRUE
  for(data in datalist){
    
    if(do_sample){
      train_ids <- createDataPartition(data$diabetes_status, p = 0.5, list = FALSE, times = 1)
      test_ids <- setdiff(1:dim(data)[1],train_ids)
      do_sample = FALSE
    }
    if(accidx==5){
      accidx = accidx + 1
      next
    }
    print(i)
    # print(accidx)
    # print(train_ids)
    
    train_data <- data[train_ids, ]
    test_data  <- data[test_ids, ]
    
    # print('base')
    classifier_base <- randomForest(diabetes_status ~ age + sex + bmi.numeric, family='binomial', data=train_data)
    predClass <- predict(classifier_base, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    # print(cm)
    cm <- cm_sanity_check(cm)
    acc[i,1, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,2, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    # print('vol')
    classifier_vol <- randomForest(diabetes_status ~ seg_liver_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_vol, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    # print(cm)
    cm <- cm_sanity_check(cm)
    acc[i,3, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,4, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    
    # print('iou')
    classifier_iou <- randomForest(diabetes_status ~ seg_liver_scaled + iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_iou, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    # print(cm)
    cm <- cm_sanity_check(cm)
    acc[i,5, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,6, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    # print('cvinv')
    classifier_cvinv <- randomForest(diabetes_status ~ seg_liver_scaled + cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_cvinv, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    # print(cm)
    cm <- cm_sanity_check(cm)
    acc[i,7, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,8, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    # print('instance iou')
    classifier_instanceiou <- randomForest(diabetes_status ~ seg_liver_scaled, weights = train_data$iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_instanceiou, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    # print(cm)
    cm <- cm_sanity_check(cm)
    acc[ i,9, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,10, accidx]<- 2 * ((prec * rec) / (prec + rec))
    
    # print('sample iou')
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
    # print(cm)
    cm <- cm_sanity_check(cm)
    acc[ i,11, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,12, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    # print('instance cvinv')
    classifier_instancecvinv <- randomForest(diabetes_status ~ seg_liver_scaled, weights = train_data$cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_instancecvinv, test_data, type = "response")
    cm <- table(test_data$diabetes_status, predClass)
    # print(cm)
    cm <- cm_sanity_check(cm)
    acc[ i,13, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,14, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    # print('sample instance cvinv')
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
    # print(cm)
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

write.csv(final_mean_accs, '~/Jyotirmay/my_thesis/classification_rf_ukb_separately.csv')





