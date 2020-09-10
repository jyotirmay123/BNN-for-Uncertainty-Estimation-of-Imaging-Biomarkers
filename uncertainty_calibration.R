
require(caret)

mcdata <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/my_thesis/projects/MC_dropout_quicknat/reports/MC_dropout_quicknat_KORA_v2/KORA/10_1572006141.7793334_concat_report_final.csv')
fbdata <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/my_thesis/projects/full_bayesian/reports/full_bayesian_KORA_v4/KORA/10_1572514598.527084_concat_report_final.csv')
pbdata <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/my_thesis/projects/probabilistic_quicknat/reports/probabilistic_quicknat_KORA_v2/KORA/10_1571996796.7963011_concat_report_final.csv')
hqdata <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/my_thesis/projects/hierarchical_quicknat/reports/hierarchical_quicknat_KORA_v2/KORA/10_1571905560.9377904_concat_report_final.csv')
manualdata <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/my_thesis/dataset_groups/whole_body_datasets/KORA/all_processed_True_concat_report_final.csv')


# ggplot(mcdata, aes(x=iou_liver, y=dice_liver)) + 
#   barplot(stat="identity", color="black", position=position_dodge())

# mcdata_ = tapply(dice_liver, mcdata, sum)
hist(x = mcdata$iou_liver, y = mcdata$dice_liver)