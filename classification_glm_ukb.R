
require(caret)
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

predictor <- function(classifier, test_data){
  predClass <- predict(classifier, test_data, type = "response")
  cm <- table(test_data$diabetes_status, predClass>0.5)
  cm <- cm_sanity_check(cm)
  accuracy <- sum(diag(cm)) / sum(cm)
  prec <- precision(cm)
  rec <- recall(cm)
  f1 <- 2 * ((prec * rec) / (prec + rec))
  quality_measures <- list("accuracy" = accuracy, "f1" = f1)
  return (quality_measures)
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
ukbprocessedlist = list(mcukbdata, fbukbdata, pbukbdata, hqukbdata, manualukbdata)

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
final_mean_accs <- array(0, dim=c(32, 5))
acc <- array(0, dim=c(freq, 32, 5))

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
    
    train_data <- data[train_ids, ]
    test_data  <- data[test_ids, ]
    ukb_data <- get_appropriate_ukb_dataset(accidx)
    
    classifier_base <- glm(diabetes_status ~ age + sex + bmi.numeric, family='binomial', data=train_data)
    measures <- predictor(classifier_base, test_data)
    acc[i,1, accidx] <- measures$accuracy
    acc[i,2, accidx] <- measures$f1
    
    measures <- predictor(classifier_base, ukb_data)
    acc[i,17, accidx] <- measures$accuracy
    acc[i,18, accidx] <- measures$f1
    
    classifier_vol <- glm(diabetes_status ~ seg_liver_scaled, family='binomial', data=train_data)
    measures <- predictor(classifier_vol, test_data)
    acc[i,3, accidx] <- measures$accuracy
    acc[i,4, accidx] <- measures$f1
    
    measures <- predictor(classifier_vol, ukb_data)
    acc[i,19, accidx] <- measures$accuracy
    acc[i,20, accidx] <- measures$f1
    
    if(accidx==5){
      accidx = accidx + 1
      next
    }
    
    classifier_iou <- glm(diabetes_status ~  seg_liver_scaled + iou_liver, family='binomial', data=train_data)
    measures <- predictor(classifier_iou, test_data)
    acc[i,5, accidx] <- measures$accuracy
    acc[i,6, accidx] <- measures$f1
    
    measures <- predictor(classifier_iou, ukb_data)
    acc[i,21, accidx] <- measures$accuracy
    acc[i,22, accidx] <- measures$f1
    
    classifier_cvinv <- glm(diabetes_status ~  seg_liver_scaled + cvinv_scaled, family='binomial', data=train_data)
    measures <- predictor(classifier_cvinv, test_data)
    acc[i,7, accidx] <- measures$accuracy
    acc[i,8, accidx] <- measures$f1
    
    measures <- predictor(classifier_cvinv, ukb_data)
    acc[i,23, accidx] <- measures$accuracy
    acc[i,24, accidx] <- measures$f1
    
    classifier_instanceiou <- glm(diabetes_status ~ seg_liver_scaled, weights = train_data$iou_liver, family='binomial', data=train_data)
    measures <- predictor(classifier_instanceiou, test_data)
    acc[i,9, accidx] <- measures$accuracy
    acc[i,10, accidx] <- measures$f1
    
    measures <- predictor(classifier_instanceiou, ukb_data)
    acc[i,25, accidx] <- measures$accuracy
    acc[i,26, accidx] <- measures$f1
    
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
    
    predClass <- colMeans(pc) > 0.5
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[ i,11, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,12, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    pc <- array(0, dim=c(10, nrow(ukb_data)))
    pc[1,] <- sample_predictor_analyser(ukb_data$X0_liver_scaled, ukb_data, classifier_instanceiou)
    pc[2,] <- sample_predictor_analyser(ukb_data$X1_liver_scaled, ukb_data, classifier_instanceiou)
    pc[3,] <- sample_predictor_analyser(ukb_data$X2_liver_scaled, ukb_data, classifier_instanceiou)
    pc[4,] <- sample_predictor_analyser(ukb_data$X3_liver_scaled, ukb_data, classifier_instanceiou)
    pc[5,] <- sample_predictor_analyser(ukb_data$X4_liver_scaled, ukb_data, classifier_instanceiou)
    pc[6,] <- sample_predictor_analyser(ukb_data$X5_liver_scaled, ukb_data, classifier_instanceiou)
    pc[7,] <- sample_predictor_analyser(ukb_data$X6_liver_scaled, ukb_data, classifier_instanceiou)
    pc[8,] <- sample_predictor_analyser(ukb_data$X7_liver_scaled, ukb_data, classifier_instanceiou)
    pc[9,] <- sample_predictor_analyser(ukb_data$X8_liver_scaled, ukb_data, classifier_instanceiou)
    pc[10,] <- sample_predictor_analyser(ukb_data$X9_liver_scaled, ukb_data, classifier_instanceiou)
    
    predClass <- colMeans(pc) > 0.5
    cm <- table(ukb_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[ i,27, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,28, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    classifier_instancecvinv <- glm(diabetes_status ~  seg_liver_scaled, weights = train_data$cvinv_scaled, family='binomial', data=train_data)
    measures <- predictor(classifier_instancecvinv, test_data)
    acc[i,13, accidx] <- measures$accuracy
    acc[i,14, accidx] <- measures$f1
    
    measures <- predictor(classifier_instancecvinv, ukb_data)
    acc[i,29, accidx] <- measures$accuracy
    acc[i,30, accidx] <- measures$f1
    
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
    
    predClass <- colMeans(pc) > 0.5
    cm <- table(test_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[ i,15, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,16, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    pc <- array(0, dim=c(10, nrow(ukb_data)))
    pc[1,] <- sample_predictor_analyser(ukb_data$X0_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[2,] <- sample_predictor_analyser(ukb_data$X1_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[3,] <- sample_predictor_analyser(ukb_data$X2_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[4,] <- sample_predictor_analyser(ukb_data$X3_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[5,] <- sample_predictor_analyser(ukb_data$X4_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[6,] <- sample_predictor_analyser(ukb_data$X5_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[7,] <- sample_predictor_analyser(ukb_data$X6_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[8,] <- sample_predictor_analyser(ukb_data$X7_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[9,] <- sample_predictor_analyser(ukb_data$X8_liver_scaled, ukb_data, classifier_instancecvinv)
    pc[10,] <- sample_predictor_analyser(ukb_data$X9_liver_scaled, ukb_data, classifier_instancecvinv)
    
    predClass <- colMeans(pc) > 0.5
    cm <- table(ukb_data$diabetes_status, predClass)
    cm <- cm_sanity_check(cm)
    acc[ i,31, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    acc[i,32, accidx] <- 2 * ((prec * rec) / (prec + rec))
    
    accidx = accidx + 1
  }
}
final_mean_accs[,1] = colMeans(acc[1:freq,,1])
final_mean_accs[,2] = colMeans(acc[1:freq,,2])
final_mean_accs[,3] = colMeans(acc[1:freq,,3])
final_mean_accs[,4] = colMeans(acc[1:freq,,4])
final_mean_accs[,5] = colMeans(acc[1:freq,,5])

final_mean_accs <- aperm(final_mean_accs)

write.csv(final_mean_accs, '~/Jyotirmay/my_thesis/classification_glm_ukb_volume_only_all.csv')





