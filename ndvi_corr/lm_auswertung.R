ml <- readRDS("/home/lukas/Documents/ETH/MASTER_THESIS/code/data/computation_results/ml_models/R/ml_lm_lm_ndvi_scl_stats_ss_noex.rds", refhook = function(x) NULL)
ml_res <- readRDS("/home/lukas/Documents/ETH/MASTER_THESIS/code/data/computation_results/ml_models/R/ml_lm_lm_ndvi_scl_res_stats_ss_noex.rds", refhook = function(x) NULL)

for (M in list(ml, ml_res)) {
    print(summary(M))
    a <- M$coefficients
    a <- c(a[2], a[1], a[3:length(a)] + a[1])
    names(a) <- c("NDVI", "SCL2", "SCL3", "SCL4", "SCL5", "SCL6", "SCL7", "SCL8", "SCL9", "SCL10", "SCL11")
    print((round(as.matrix(a), 3)))
}
