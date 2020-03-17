
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
final_mean_accs <- array(0, dim=c(20, 5))
acc <- array(0, dim=c(freq, 20, 5))

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

    classifier_base <- glm(diabetes_status ~ age + sex + bmi.numeric, family='binomial', data=train_data)
    predClass <- predict(classifier_base, test_data, type = "response")
    acc[i,2, accidx]<-  AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[i,1, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
     #2 * ((prec * rec) / (prec + rec))

    classifier_vol <- glm(diabetes_status ~ seg_liver_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_vol, test_data, type = "response")
    acc[i,4, accidx]<-  AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[i,3, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
     #2 * ((prec * rec) / (prec + rec))

    if(accidx==5){
      accidx = accidx + 1
      next
    }

    classifier_iou <- glm(diabetes_status ~ seg_liver_scaled + iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_iou, test_data, type = "response")
    acc[i,6, accidx]<-  AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[i,5, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
   # 2 * ((prec * rec) / (prec + rec))

    classifier_cvinv <- glm(diabetes_status ~ seg_liver_scaled + cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_cvinv, test_data, type = "response")
    acc[i,8, accidx]<- AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[i,7, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
     # 2 * ((prec * rec) / (prec + rec))
    
    classifier_instanceiou <- glm(diabetes_status ~ seg_liver_scaled, weights = train_data$iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_instanceiou, test_data, type = "response")
    acc[i,10, accidx]<- AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[ i,9, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
     # 2 * ((prec * rec) / (prec + rec))
    
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
    
    predClass <- colMeans(pc)
    acc[i,12, accidx] <- AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[ i,11, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
     # 2 * ((prec * rec) / (prec + rec))
    
    classifier_instancecvinv <- glm(diabetes_status ~ seg_liver_scaled, weights = train_data$cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_instancecvinv, test_data, type = "response")
    acc[i,14, accidx] <-  AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[ i,13, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
     #2 * ((prec * rec) / (prec + rec))
  
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
    
    predClass <- colMeans(pc)
    acc[i,16, accidx] <-  AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[ i,15, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
     #2 * ((prec * rec) / (prec + rec))
    
    classifier_iou <- glm(diabetes_status ~ seg_liver_scaled * iou_liver, family='binomial', data=train_data)
    predClass <- predict(classifier_iou, test_data, type = "response")
    acc[i,18, accidx]<-  AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[i,17, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    # 2 * ((prec * rec) / (prec + rec))
    
    classifier_cvinv <- glm(diabetes_status ~ seg_liver_scaled * cvinv_scaled, family='binomial', data=train_data)
    predClass <- predict(classifier_cvinv, test_data, type = "response")
    acc[i,20, accidx]<- AUC( predClass, test_data$diabetes_status)
    cm <- table(test_data$diabetes_status, predClass>0.5)
    cm <- cm_sanity_check(cm)
    acc[i,19, accidx] <- sum(diag(cm)) / sum(cm)
    prec <- precision(cm)
    rec <- recall(cm)
    
    accidx = accidx + 1
  }
}
final_mean_accs[,1] = colMeans(acc[1:freq,,1])
final_mean_accs[,2] = colMeans(acc[1:freq,,2])
final_mean_accs[,3] = colMeans(acc[1:freq,,3])
final_mean_accs[,4] = colMeans(acc[1:freq,,4])
final_mean_accs[,5] = colMeans(acc[1:freq,,5])

final_mean_accs <- aperm(final_mean_accs)

write.csv(final_mean_accs, '~/Jyotirmay/my_thesis/glm_classification_kora_auc_sample_analyzer_interactions.csv')




# liv_samp <- as.matrix(mcdata[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
# cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
# mcdata$cvinv = 1/cv
# 
# liv_samp <- as.matrix(fbdata[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
# cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
# fbdata$cvinv = 1/cv
# 
# liv_samp <- as.matrix(pbdata[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
# cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
# pbdata$cvinv = 1/cv
# 
# liv_samp <- as.matrix(hqdata[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
# cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
# hqdata$cvinv = 1/cv
# 
# 
# aa <- cbind(mcdata$iou_liver,rep(1,153))
# bb <- cbind(fbdata$iou_liver,rep(2,153))
# cc <- cbind(pbdata$iou_liver,rep(3,153))
# dd <- cbind(hqdata$iou_liver,rep(4,153))
# aa <- cbind(rep('Manual', 153), manualdata$diabetes_status, manualdata$seg_liver)


# aa <- cbind(mcdata$cvinv,rep(1,153))
# bb <- cbind(fbdata$cvinv,rep(2,153))
# cc <- cbind(pbdata$cvinv,rep(3,153))
# dd <- cbind(hqdata$cvinv,rep(4,153))
# 
# comb <- rbind(aa,bb,cc,dd)
# 
# comb <- as.data.frame(comb)
# 
# colnames(comb) <- c("iou","net")
# comb$net <- as.factor(comb$net)
# ggplot(comb, aes(x = iou, fill = net)) + geom_density(alpha = 0.5, adjust=5)

# Plotting liver volume box plot for diabetic and non diabetic across models.
aa <- cbind(rep('Manual', 153), manualdata$diabetes_status, manualdata$, rep(0, 153))
bb <- cbind(rep("MC Dropout", 153), mcdata$diabetes_status, mcdata$seg_liver, mcdata$dice_liver)
cc <- cbind(rep("Bayesian", 153), fbdata$diabetes_status, fbdata$seg_liver, fbdata$dice_liver)
dd <- cbind(rep("Probabilistic", 153), pbdata$diabetes_status, pbdata$seg_liver, pbda)
ee <- cbind(rep("Hierarchical", 153), hqdata$diabetes_status, hqdata$seg_liver)
tmp <- rbind(aa,bb,cc,dd,ee)
tmp <- as.data.frame(tmp)
colnames(tmp) <- c('model', 'diabetes_status', 'liver_volume')
tmp$diabetes_status <- as.numeric(levels(tmp$diabetes_status))[tmp$diabetes_status]
tmp$diabetes_status[tmp$diabetes_status == 2] <- 1
tmp$diabetes_status <- as.factor(tmp$diabetes_status)
tmp$liver_volume <- as.numeric(levels(tmp$liver_volume))[tmp$liver_volume]
ggplot(tmp, aes(x=model, y=liver_volume, fill=diabetes_status)) +
geom_boxplot() + theme(legend.position="bottom")


aa <- cbind(rep('Manual', 153), manualdata$diabetes_status, manualdata, rep(0, 153))
bb <- cbind(rep("MC Dropout", 1), rep(mean(mcdata$dice_liver), 1), rep(sd(mcdata$dice_liver), 1))
cc <- cbind(rep("Bayesian", 1), rep(mean(fbdata$dice_liver), 1), rep(sd(mcdata$dice_liver), 1))
dd <- cbind(rep("Probabilistic", 1), rep(mean(pbdata$dice_liver), 1), rep(sd(mcdata$dice_liver), 1))
ee <- cbind(rep("Hierarchical", 1), rep(mean(hqdata$dice_liver), 1), rep(sd(mcdata$dice_liver), 1))
tmp <- rbind(bb,cc,dd,ee)
tmp <- as.data.frame(tmp)
colnames(tmp) <- c('model', 'liver_volume', 'sd')
# tmp$diabetes_status <- as.numeric(levels(tmp$diabetes_status))[tmp$diabetes_status]
# tmp$diabetes_status[tmp$diabetes_status == 2] <- 1
# tmp$diabetes_status <- as.factor(tmp$diabetes_status)
tmp$liver_volume <- as.numeric(levels(tmp$liver_volume))[tmp$liver_volume]
tmp$sd <- as.numeric(levels(tmp$sd))[tmp$sd]
# ggplot(tmp, aes(x=model, y=liver_volume, fill=diabetes_status)) +
#   geom_boxplot() + theme(legend.position="bottom")


ggplot(tmp, aes(x=model, y=liver_volume)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=liver_volume-sd, ymax=liver_volume+sd), width=.2,
                position=position_dodge(.9)) + ylim(0.8, 1.0)
