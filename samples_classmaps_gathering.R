# KORA2452967
path <- '~/home/abhijit/Jyotirmay/my_thesis/projects/MC_dropout_quicknat/outs/MC_dropout_quicknat_KORA_v2/MC_dropout_quicknat_KORA_v2_predictions_KORA/KORA2452967_samples'

files <- list.files(path="~/Jyotirmay/my_thesis/projects/MC_dropout_quicknat/outs/MC_dropout_quicknat_KORA_v2/MC_dropout_quicknat_KORA_v2_predictions_KORA/KORA2452967_samples/KORA2452967_sample_0.nii.gz", pattern="*.nii.gz", full.names=TRUE, recursive=FALSE)
print(files)

s_path <- "~/Jyotirmay/my_thesis/projects/MC_dropout_quicknat/outs/MC_dropout_quicknat_KORA_v2/MC_dropout_quicknat_KORA_v2_predictions_KORA/KORA2452967_samples/KORA2452967_sample_0.nii.gz"
if (!"devtools" %in% installed.packages()[, "Package"]) {
  install.packages("devtools")
}
if (!"ms.lesion" %in% installed.packages()[, "Package"]) {
  devtools::install_github("muschellij2/ms.lesion")
}
source("https://neuroconductor.org/neurocLite.R")
pkgs = c("neurobase", "fslr", "dcm2niir","divest", 
         "RNifti", "oro.dicom", 
         "oro.nifti", "WhiteStripe", "neurohcp", "papayar",
         "papayaWidget", "oasis", "kirby21.t1")
neuro_install(pkgs)
# data for whitestripe
library(dcm2niir); install_dcm2nii()
library(WhiteStripe); download_img_data()