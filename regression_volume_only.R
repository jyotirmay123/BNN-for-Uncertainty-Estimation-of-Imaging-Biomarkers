
get_regression_stat <- function(test_data_csv) {
  #~/Desktop/datas/MCD/10_1572006141.7793334_concat_report_final.csv"
  tab <- read.csv(test_data_csv)
  tab_man <- read.csv("~/Desktop/datas/all_processed_True_concat_report_final.csv")
  tab_man$man_liver <- tab_man$seg_liver
  
  liv_samp <- as.matrix(tab[,c("X0_liver","X1_liver","X2_liver","X3_liver","X4_liver","X5_liver","X6_liver","X7_liver","X8_liver","X9_liver")])
  rowMeans(liv_samp)
  #rowSds(liv_samp)
  
  SD=apply(liv_samp,1, sd, na.rm = TRUE)
  tab$cv <- apply(liv_samp,1, sd, na.rm = TRUE) /  rowMeans(liv_samp)
  tab$cv_inv = 1/tab$cv
  
  tab <- merge(tab, tab_man[,c("volume_id","man_liver")], by="volume_id")
  
  abc <- ''
  lm_fit_manual <- lm(scale(man_liver) ~  diabetes_status, data=tab) #0.000149
  abc$mcd_manual <- summary(lm_fit_manual)[['coefficients']]
  
  lm_fit_base <- lm(scale(seg_liver) ~  diabetes_status, data=tab) #0.000483
  abc$mcd_base <- summary(lm_fit_base)[['coefficients']]
  
  lm_fit_iou <- lm(scale(seg_liver) ~  diabetes_status + iou_liver, data=tab) #0.000347
  abc$mcd_iou <- summary(lm_fit_iou)[['coefficients']]
  
  # lm_fit_cv <- lm(scale(seg_liver) ~  diabetes_status + cv, data=tab) #0.000449
  # abc$mcd_cv <- summary(lm_fit_cv)[['coefficients']]
  
  lm_fit_cvinv <- lm(scale(seg_liver) ~  diabetes_status + cv_inv, data=tab) #0.000278
  abc$mcd_cvinv <- summary(lm_fit_cvinv)[['coefficients']]
  
  lm_fit_instanceiou <- lm(scale(seg_liver) ~  diabetes_status, weight=iou_liver, data=tab) #0.000274
  abc$mcd_instanceiou  <- summary(lm_fit_instanceiou)[['coefficients']]
  
  lm_fit_instancecvinv <- lm(scale(seg_liver) ~  diabetes_status, weight=cv_inv, data=tab) #0.00335
  abc$mcd_instancecvinv <- summary(lm_fit_instancecvinv)[['coefficients']]
  
  return(abc)
}

