library("scmamp")
library("ggplot2")
library("Rgraphviz")

data_mc <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/all_accs/mc_acc.csv')
data_fb <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/all_accs/fb_acc.csv')
data_pb <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/all_accs/pb_acc.csv')
data_hq <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/all_accs/hq_acc.csv')


comb <- cbind(data_mc,data_fb,data_pb,data_hq)

write.csv(comb, '/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/all_accs/all_acc.csv')

data <- read.csv('/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/all_accs/all_acc.csv')

# data <- as.data.frame(comb)

data <- data

plotDensities(data=data, size=1.1)


qqplotGaussian(data[, ], size=5 , col="orchid") + theme_classic()


test<-nemenyiTest(data, alpha=0.05)
outs <- as.data.frame(test$diff.matrix)

abs(test$diff.matrix) > test$statistic

# write.csv(outs, '/Users/jyotirmaysenapati/Desktop/Projects/PYTHON/all_stats_out.csv')

plotCD(data, alpha=0.05, cex=1.25)


friedmanAlignedRanksPost(data, control = "Base")
