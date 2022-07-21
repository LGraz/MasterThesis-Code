require(randomForest)
require("reticulate") # to load pyhton data frame
source("my_utils/fun_to_get_covariates.r")


verbose <- FALSE

# load data
source_python("my_utils/R_read_pkl.py")
dir_name <- "./data/computation_results/pixels_itpl_corr_dict_array/"
max_file <- as.integer(gsub(".pkl", "", list.files(dir_name)))
lists <- load_pickle(paste0(dir_name, as.character(max_file), ".pkl"))

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
    NDVI_ITPL_DATA[[pix]][["gdd"]] <- (lists[[pix]]$gdd)
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
get_table_row_with_covariates_and_yield <- function(ndvi_ts, yield, gdd_ts) {
    covariates <- sapply(fun_to_get_covariates, function(fun) fun(gdd_ts, ndvi_ts))
    c(yield = unname(yield), covariates)
}
grid <- as.matrix(sapply(expand.grid(c(pix = list(seq_along(lists)), template_names)), as.character))
invisible(apply(grid, 1, function(x) {
    ndvi_ts <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$itpl[[x["strat"], x["itpl_meth"], x["short_names"]]]
    yield <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$yield
    gdd_series <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$gdd
    array_for_estimation[x["pix"], x["strat"], x["itpl_meth"], x["short_names"], ] <<- get_table_row_with_covariates_and_yield(
        ndvi_ts, yield, gdd_series
    )
    if (runif(1) < 0.01) {
        cat(".")
    }
}))
# for (i in 1:nrow(grid)) {
#     x <- grid[i, ]
#     ndvi_ts <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$itpl[[x["strat"], x["itpl_meth"], x["short_names"]]]
#     yield <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$yield
#     gdd <- NDVI_ITPL_DATA[[as.integer(x["pix"])]]$gdd
#     array_for_estimation[x["pix"], x["strat"], x["itpl_meth"], x["short_names"], ] <- get_table_row_with_covariates_and_yield(
#         ndvi_ts, yield, gdd
#     )
# }
if (verbose) {
    str(array_for_estimation)
}


##############################################################
###   get prediction model
##############################################################
from_data_to_evaluation <- function(data) {
    set.seed(4321)
    data <- as.data.frame(data)
    rf <- randomForest(yield ~ ., data, ntree = 5)
    predicted <- rf$predicted # may contain NA'a (if observation considered in every tree)
    rm(rf)
    residuals <- data$yield - predicted
    statistics <- list(
        rmse = function(res) sqrt(mean(res^2, na.rm = TRUE))
    )
    sapply(statistics, function(fun) fun(residuals))
}

# apply prediction model
data <- array_for_estimation[, 1, 1, 1, ]
model_array <- apply(array_for_estimation, c(2, 3, 4), from_data_to_evaluation)
str(model_array, max.level = 1)
dimnames(model_array)

# summary of yield
summary(yield <- sapply(NDVI_ITPL_DATA, function(l) l$yield))



a <- model_array["id", , ]
b <- model_array["rob", , ]
rownames(b) <- paste0(rownames(b), "_rob")
results_df <- rbind(a, b)
dimnames(results_df) <- lapply(dimnames(results_df), function(x) gsub("_", "-", x))
results_df

# write to latex
source_python("my_utils/plot_colored_pandas_df.py")


write_df_to_latex(as.data.frame(results_df), "../latex/tex/chapters/misc/table_methods_vs_yieldprediction.tex")
