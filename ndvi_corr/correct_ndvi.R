# load data
require("reticulate")
path <- "data/computation_results/ndvi_tables/ndvi_table_0.01"
source_python("my_utils/R_read_pkl.py")
ndvi_df <- read_pickle_file("data/computation_results/ndvi_tables/ndvi_table_0.01")
ndvi_df$scl_class <- as.factor(ndvi_df$scl_class)
str(ndvi_df)

response <- c(
    "ndvi_itpl_ss_noex"
    # "ndvi_itpl_loess_noex"
    # "ndvi_itpl_dl"
    # "ndvi_itpl_ss_noex_rob_rew_1"
    # "ndvi_itpl_loess_noex_rob_rew_1"
    # "ndvi_itpl_dl_rob_rew_1"
)
covariates <- c(
    "ndvi_observed",
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    "cum_rain",
    "avg_temp",
    "day_rain",
    "max_temp",
    "min_temp",
    "scl_class"
)
full_formula <- as.formula(paste(response, " ~ ", paste(covariates, collapse = "+")))


# linear model ----------------------------------
lm_ndvi_scl <- lm(as.formula(paste(response, " ~ ", "ndvi_observed + scl_class")), data = ndvi_df)
summary(lm_ndvi_scl)
lm_all <- lm(full_formula, data = ndvi_df)
summary(lm_all)
op <- par(mfrow = c(2, 2))
plot(lm_all)
par(op)
lm_step_selected <- step(lm_all, )


# random Forest
require(randomForest)
forest <- randomForest(full_formula, data = ndvi_df, ntree = 5)
summary(forest)
str(forest)
