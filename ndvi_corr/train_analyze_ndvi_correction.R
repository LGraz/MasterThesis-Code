source("./my_utils/R_help_fun.R")

require("reticulate") # to load pyhton data frame

verbose <- FALSE
update <- FALSE
clean_before <- FALSE
unique_name <- FALSE
if (clean_before) {
  for (f in list.files("data/computation_results/ml_models/R/", full.names = TRUE)) {
    file.remove(f)
  }
}

# load data
path <- "data/computation_results/ndvi_tables/ndvi_table_all_years_001.pkl"
source_python("my_utils/R_read_pkl.py")
ndvi_df <- read_pickle_file("data/computation_results/ndvi_tables/ndvi_table_all_years_001.pkl")
ndvi_df$scl_class <- factor(ndvi_df$scl_class, seq(2, 11))
if (verbose) {
  str(ndvi_df)
}

responses <- c(
  "ndvi_itpl_ss_noex",
  # "ndvi_itpl_loess_noex",
  "ndvi_itpl_dl",
  "ndvi_itpl_ss_noex_rob_rew_1",
  # "ndvi_itpl_loess_noex_rob_rew_1",
  "ndvi_itpl_dl_rob_rew_1"
)
covariates_no_scl <- c(
  # "cum_rain",
  # "avg_temp",
  # "day_rain",
  # "max_temp",
  # "min_temp",
  # "scl_class",
  "ndvi_observed",
  "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
)
covariates <- c(covariates_no_scl, "scl_class")



require(randomForest)
require(earth)
require(mgcv)
require(caret) # nolint

# Define ML methods here
get_models_for_response <- function(response) {
  ###################################################
  ####    Help Functions First
  ###################################################
  get_res <- function() get("response_res", envir = fun_inner_env)

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
    cat("load/generate:", string, "\n")
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

  load_or_generate <- function(name, package, predict_suffix, response, body_basic) {
    name <- paste(predict_suffix, name, sep = "_")
    # obj, name, response, ndvi_df, covariates, package, predict_suffix, load_instead = FALSE
    obj <- RDS_save_or_load(NULL, name, response, ndvi_df, covariates, package, predict_suffix, load_instead = TRUE)
    if (is.null(obj) || update) { # nolint
      obj <- eval(body_basic)
      RDS_save_or_load(obj, name, response, ndvi_df, covariates, package, predict_suffix) # nolint
    }
    models[name] <<- list(obj)
    if (verbose) {
      summary(obj)
    }
  }

  fun_inner_env <- NULL
  load_or_generate_both <- function(name, package, predict_suffix, response, body_basic = NULL, body_corr = NULL) {
    fun_inner_env <<- environment()
    load_or_generate(name, package, predict_suffix, response, enquote(body_basic))
    # get residuals
    model <- models[[paste(predict_suffix, name, sep = "_")]]
    response_res <- paste0(response, "_res")
    ndvi_df[response_res] <<- abs(ndvi_df[response] - predict(model, ndvi_df))
    # ndvi_df[response_res] <- residuals(model)
    load_or_generate(paste0(name, "_res"), package, predict_suffix, response, enquote(body_corr))
  }

  ###################################################
  ####    NOW TRAIN MODELS
  ###################################################
  ndvi_df <- ndvi_df
  models <- vector("list")
  # linear model ----------------------------------
  paste(paste0(response, "_res"), " ~ ", "ndvi_observed + scl_class")
  load_or_generate_both(
    "ndvi_scl", "stats", "lm", response,
    body_basic = lm((paste(response, " ~ ", "ndvi_observed + scl_class")), data = ndvi_df),
    body_corr = lm((paste(get_res(), " ~ ", "ndvi_observed + scl_class")), data = ndvi_df)
  )


  full_formula <- as.formula(paste(response, " ~ ", paste(covariates, collapse = "+")))
  full_formula1 <- as.formula(paste(get_res(), " ~ ", paste(covariates, collapse = "+")))
  load_or_generate_both(
    "all", "stats", "lm", response,
    lm(full_formula, data = ndvi_df),
    lm(full_formula1, data = ndvi_df)
  )

  # random Forest ----------------------------------
  # require(randomForest)
  ntree <- 100
  set.seed(4321)
  load_or_generate_both(
    "rf", "randomForest", "randomForest", response,
    randomForest(full_formula, data = ndvi_df, ntree = ntree),
    randomForest(full_formula1, data = ndvi_df, ntree = ntree)
  )

  # M A R S -------------------------------
  # require(earth)
  load_or_generate_both(
    "mars", "earth", "earth", response,
    earth(full_formula, ndvi_df, degree = 2),
    earth(full_formula1, ndvi_df, degree = 2)
  )

  # G A M ----------------------------------
  # require(mgcv)
  full_gam_formula <- as.formula(paste(response, " ~ ", paste(paste0("s(", covariates_no_scl, ")"), collapse = "+")))
  full_gam_formula1 <- as.formula(paste(get_res(), " ~ ", paste(paste0("s(", covariates_no_scl, ")"), collapse = "+")))
  load_or_generate_both(
    "gam", "mgcv", "gam", response,
    gam(full_gam_formula, data = ndvi_df),
    gam(full_gam_formula1, data = ndvi_df)
  )

  # lasso ------------------------------------
  # require(caret)
  full_lasso_formula <- as.formula(paste(response, " ~ (", paste(covariates, collapse = "+"), ")^2"))
  full_lasso_formula1 <- as.formula(paste(get_res(), " ~ (", paste(covariates, collapse = "+"), ")^2"))
  set.seed(4321)
  load_or_generate_both(
    "lasso",
    "caret",
    "train",
    response,
    {
      # specifying the CV technique which will be passed into the train() function later and number parameter is the "k" in K-fold cross validation
      train_control <- caret::trainControl(method = "cv", number = 5, search = "grid")
      ## Customsing the tuning grid (lasso regression has alpha = 1)
      lassoGrid <- expand.grid(alpha = 1, lambda = c(2^seq(-20, 5, length = 50)))
      # training a Lasso Regression model while tuning parameters
      caret::train(full_lasso_formula, data = ndvi_df, method = "glmnet", trControl = train_control, tuneGrid = lassoGrid, relax = TRUE)
    },
    {
      # specifying the CV technique which will be passed into the train() function later and number parameter is the "k" in K-fold cross validation
      train_control <- caret::trainControl(method = "cv", number = 5, search = "grid")
      ## Customsing the tuning grid (lasso regression has alpha = 1)
      lassoGrid <- expand.grid(alpha = 1, lambda = c(2^seq(-20, 5, length = 50)))
      # training a Lasso Regression model while tuning parameters
      caret::train(full_lasso_formula1, data = ndvi_df, method = "glmnet", trControl = train_control, tuneGrid = lassoGrid, relax = TRUE)
    }
  )

  # # lm - stepwise selection
  # load_or_generate_both(
  #   "step",
  #   "stats",
  #   "lm",
  #   response,
  #   {
  #     fit <- lm(full_lasso_formula, ndvi_df)
  #     step(fit, k = log(nrow(ndvi_df)), direction="backward", trace=0)
  #   },
  #   {
  #     fit <- lm(full_lasso_formula1, ndvi_df)
  #     step(fit, k = log(nrow(ndvi_df)), direction="backward", trace=0)
  #   }
  # )
  gc()
  models
}

# Now actual train methods
require(parallel)
MODELS <- vector("list")
responses_list <- as.list(responses)
names(responses_list) <- responses
MODELS <- mclapply(responses_list, get_models_for_response, mc.cores = max(1, detectCores() - 2))

list.files("./data/computation_results/ml_models/R/")


###############################################################
###   Analysis & Plots
###############################################################
models <- MODELS$ndvi_itpl_ss_noex

# rf
rf <- models[["randomForest_rf"]]
methods(class = "randomForest")
varImpPlot(rf) # sort(importance(rf))
plot(rf)
rf

# loess
loess <- models[["train_lasso"]]
loess
class(loess)
plot(loess, xTrans = log)
which(loess$finalModel$lambda == loess$finalModel$lambdaOpt)

summary(models$earth_mars)
