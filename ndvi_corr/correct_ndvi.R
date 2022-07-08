require("reticulate") # to load pyhton data frame

verbose <- FALSE
update <- FALSE
unique_name <- FALSE

# load data
path <- "data/computation_results/ndvi_tables/ndvi_table_0.01"
source_python("my_utils/R_read_pkl.py")
ndvi_df <- read_pickle_file("data/computation_results/ndvi_tables/ndvi_table_0.01")
ndvi_df$scl_class <- as.factor(ndvi_df$scl_class)
if (verbose){
  str(ndvi_df)
}

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

RDS_save_or_load <- function(obj, name, response, ndvi_df, covariates, package, predict_suffix, load_instead = FALSE) {
  if (unique_name) {
    string <- paste("ml", predict_suffix, name, package, gsub("ndvi_itpl_", "", response),
      nrow(ndvi_df), paste0(sapply(covariates, substr, 1, 1), collapse = ""),
      mean(ndvi_df$ndvi_observed),
      sep = "_"
    )
  } else {
    string <- paste("ml", predict_suffix, name, package, gsub("ndvi_itpl_", "", response), sep = "_")
  }
  if (length(string) != 1) {
    if (verbose) {
      print("_________________________________________________")
      print(predict_suffix)
      print(package)
      print(covariates)
      print(response)
    }
    stop("filename not one-dim")
  }
  fname <- paste0("./data/computation_results/ml_models/R/", string, ".rds")
  if (load_instead) {
    if (file.exists(fname)) {
      if (verbose) {
        print("loads file --------------")
      }
      obj <- readRDS(fname)
    } else {
      if (verbose) {
        print("file did not exist ------------")
        print(getwd())
      }
      return(NULL)
    }
  }
  saveRDS(obj, fname)
  obj
}
load_or_generate <- function(name, package, predict_suffix, response, body) {
  name <- paste(predict_suffix, name, sep = "_")
  # obj, name, response, ndvi_df, covariates, package, predict_suffix, load_instead = FALSE
  obj <- RDS_save_or_load(NULL, name, response, ndvi_df, covariates, package, predict_suffix, load_instead = TRUE)
  if (is.null(obj) || update) { # nolint
    obj <- eval(substitute(body))
    RDS_save_or_load(obj, name, response, ndvi_df, covariates, package, predict_suffix) # nolint
  }
  models[name] <<- list(obj)
  if (verbose) {
    summary(obj)
  }
}

load_or_generate_both <- function(name, package, predict_suffix, response, body) {
  load_or_generate(name, package, predict_suffix, response, body)
  model_name <- paste(predict_suffix, name, sep = "_")
  response_res <- paste0(model_name, "_res")
  ndvi_df[[response_res]] <<- ndvi_df[response] - predict(models[[model_name]], ndvi_df)
  load_or_generate(paste0(name, "_res"), package, predict_suffix, response_res, body)
}

###################################################
####    NOW TRAIN MODELS
###################################################
models <- vector("list")

# linear model ----------------------------------
load_or_generate_both(
  "ndvi_scl", "stats", "lm", response,
  lm(as.formula(paste(response, " ~ ", "ndvi_observed + scl_class")), data = ndvi_df)
)

full_formula <- as.formula(paste(response, " ~ ", paste(covariates, collapse = "+")))
load_or_generate_both(
  "all", "stats", "lm", response,
  lm(full_formula, data = ndvi_df)
)

# random Forest ----------------------------------
require(randomForest)
ntree <- 10
load_or_generate_both(
  "rf", "randomForest", "randomForest", response,
  randomForest(full_formula, data = ndvi_df, ntree = ntree)
)




names(models)
