require(randomForest)
require(parallel)
require("reticulate") # to load pyhton data frame
source("my_utils/R_fun_to_get_covariates.r")


verbose <- FALSE

# load data
source_python("my_utils/R_read_pkl.py")
dir_name <- "./data/computation_results/pixels_itpl_corr_dict_array/"
max_file <- max(as.integer(gsub(".pkl", "", list.files(dir_name))))
lists <- load_pickle(paste0(dir_name, as.character(max_file), ".pkl"))
print("loaded data ---------------------------------------")
#############################################################
###  create structures for data (list & arrays & array-lists)
#############################################################

## store data in more usable format ----------------------------
#   strat x itpl_method x short_names
template_dim <- c(2, 2, length(lists[[1]][[1]][[1]]))
template_names <- list(strat = c("rob", "id"), itpl_meth = c("ss", "dl"), short_names = names(lists[[1]][[1]][[1]]))
list_template <- array(list(), dim = template_dim, dimnames = template_names)
NDVI_ITPL_DATA <- vector("list", length(lists))

# create list where each entry corresponds to its `ndvi-itpl-ts` and the `yield`
for (pix in seq_along(lists)) {
    temp <- list_template
    for (strat in c("rob", "id")) {
        for (itpl_meth in c("ss", "dl")) {
            for (short_name in names(lists[[1]][[1]][[1]])) {
                temp[[strat, itpl_meth, short_name]] <- lists[[pix]][[strat]][[itpl_meth]][[short_name]]
            }
        }
    }
    NDVI_ITPL_DATA[[pix]] <- list()
    NDVI_ITPL_DATA[[pix]][["itpl"]] <- temp
    NDVI_ITPL_DATA[[pix]][["yield"]] <- py_to_r(lists[[pix]]$yield)
    NDVI_ITPL_DATA[[pix]][["gdd"]] <- lists[[pix]]$gdd
    NDVI_ITPL_DATA[[pix]][["train_test"]] <- lists[[pix]]$train_test
}
if (verbose) {
    print("structure of NDVI_ITPL_DATA[1]  (first pixel)")
    str(NDVI_ITPL_DATA[1])
}


## array_for_estimation (covariates vs yield):   ------------------
# pix x   strat x itpl_method x short_names   x (covariates,yield)
est_dimnames <- c(
    pix = list(seq_along(lists)), template_names,
    covariates = list(c("yield", names(fun_to_get_covariates)))
)
array_for_estimation <- array(NA,
    dim = c(pix = length(lists), template_dim, length(fun_to_get_covariates) + 1),
    dimnames = est_dimnames
)
if (verbose) {
    str(array_for_estimation)
}


###########################################################
###   get covariates
###########################################################
# fill array_for_estimation with data (i.e. yield and coviariates)
require(parallel)

get_table_row_with_covariates_and_yield <- function(ndvi_ts, yield, gdd_ts) {
    covariates <- sapply(fun_to_get_covariates, function(fun) fun(gdd_ts, ndvi_ts)) # nolint
    c(yield = unname(yield), covariates)
}
grid <- as.matrix(sapply(expand.grid(c(pix = list(seq_along(lists)), template_names)), as.character))
grid_list <- lapply(as.list(1:nrow(grid)), function(x) grid[x[1], ])
n_grid <- length(grid_list)
grid_list_with_info <- vector("list", length = n_grid)
print("prepare list to get covariates --------------------")
for (i in seq_along(grid_list)) {
    x <- grid_list[[i]]
    grid_list_with_info[[i]] <- list(
        x = x,
        ndvi_ts = NDVI_ITPL_DATA[[as.integer(x["pix"])]]$itpl[[x["strat"], x["itpl_meth"], x["short_names"]]],
        yield = NDVI_ITPL_DATA[[as.integer(x["pix"])]]$yield,
        gdd_series = NDVI_ITPL_DATA[[as.integer(x["pix"])]]$gdd
    )
}

print("now get covariates  (via multicore) --------------------")
with_cov <- mclapply(grid_list_with_info, function(info) {
    if (runif(1) < 1000 / n_grid) {
        cat(".")
        gc()
    }
    try(get_table_row_with_covariates_and_yield(
        info$ndvi_ts, info$yield, info$gdd_series
    ))
}, mc.cores = max(1, round(detectCores() * 0.6)))

print("now put covariates in array")
for (i in seq_along(grid_list)) {
    x <- grid[i, ]
    if (runif(1) < 1000 / n_grid) {
        cat(".")
        gc()
    }
    array_for_estimation[x["pix"], x["strat"], x["itpl_meth"], x["short_names"], ] <- with_cov[[i]]
}
print("save array for estimation ----------------")
saveRDS(array_for_estimation, "./data/computation_results/temp_array_for_estimation.rds")

rm(list=ls())
gc()
######################################################################
###   clean environment
######################################################################

require(randomForest)
require(parallel)
require("reticulate") # to load pyhton data frame
source_python("my_utils/R_read_pkl.py")
source("my_utils/R_fun_to_get_covariates.r")
array_for_estimation <- readRDS("data/computation_results/temp_array_for_estimation.rds")

# for (i in 1:nrow(grid)) {
#     x <- grid[i, ]
#     ndvi_ts <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$itpl[[x["strat"], x["itpl_meth"], x["short_names"]]]
#     yield <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$yield
#     gdd <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$gdd
#     array_for_estimation[x["pix"], x["strat"], x["itpl_meth"], x["short_names"], ] <- get_table_row_with_covariates_and_yield(
#         ndvi_ts, yield, gdd
#     )
# }
verbose=FALSE
if (verbose) {
    str(array_for_estimation)
}



##############################################################
###   get prediction model
##############################################################

eval_stats <- list(
    rmse = function(res, data) sqrt(mean(res^2, na.rm = TRUE)),
    rrmse = function(res, data) {
        sqrt(mean(res^2, na.rm = TRUE)) / mean(data$yield)
    },
    r2 = function(res, data) {
        if (any(is.na(data$yield))) print("data yield has NA's")
        1 - (mean(res^2, na.rm = TRUE)) / (mean((data$yield - mean(data$yield, na.rm = TRUE))^2, na.rm = TRUE))
    }
)
eval_stats_posfix <- c("", "_relative", "_r2")

rename_results_df <- function(results_df){
    dimnames(results_df) <- lapply(dimnames(results_df), function(x) gsub("_", "-", x))
    colnames(results_df) <- gsub("lm", "OLS", colnames(results_df))
    colnames(results_df) <- gsub("mars", "MARS", colnames(results_df))
    colnames(results_df) <- gsub("rf", "RF", colnames(results_df))
    colnames(results_df) <- gsub("gam", "GAM", colnames(results_df))
    colnames(results_df) <- gsub("gam", "GAM", colnames(results_df))
    colnames(results_df) <- gsub("scl", "SCL", colnames(results_df))
    colnames(results_df) <- gsub("Loess", "LOESS", colnames(results_df))
    colnames(results_df) <- gsub("loess", "LOESS", colnames(results_df))
    results_df
}

for (i in 1:3) {
    from_data_to_evaluation <- function(data, i) {
        set.seed(4321)
        cat(".")
        data <- as.data.frame(data)
        rf <- randomForest(yield ~ ., data, ntree = 200)
        predicted <- rf$predicted # may contain NA'a (if observation considered in every tree)
        rm(rf)
        gc()
        residuals <- data$yield - predicted
        eval_stats[[names(eval_stats)[i]]](residuals, data)
    }

    # apply prediction model
    library(future.apply)
    plan(multicore, workers = max(1, round(detectCores() * 0.58)))
    model_array <- future_apply(
        array_for_estimation,
        c(2, 3, 4),
        from_data_to_evaluation, i,
        future.seed = TRUE
    )
    str(model_array, max.level = 1)
    dimnames(model_array)

    # # summary of yield
    # summary(yield <- sapply(NDVI_ITPL_DATA, function(l) l$yield))

    a <- model_array["id", , ]
    b <- model_array["rob", , ]
    rownames(b) <- paste0(rownames(b), "_rob")
    results_df <- rbind(a, b)
    results_df <- rename_results_df(round(results_df, digits=3))
    # write to latex
    source_python("my_utils/plot_colored_pandas_df.py")

    saveRDS(results_df, paste0("ndvi_corr/dataframe_itpl_strat", eval_stats_posfix[i], ".rds"))
    write_df_to_latex(as.data.frame(results_df),
        paste0("../latex/tex/chapters/misc/table_methods_vs_yieldprediction", eval_stats_posfix[i], ".tex")
    )
    print(results_df)
    print("done --------------------")
}
