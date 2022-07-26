
source("./my_utils/R_help_fun.R")
library(parallel)

getwd()
require("reticulate") # to load pyhton data frame
use_virtualenv("./.env/")
library(earth)
library(mgcv)
library(randomForest)
library(rlang)
library(caret)

# load data
py_config()
source_python("my_utils/R_read_pkl.py")
ndvi_df <- read_pickle_file("data/computation_results/ndvi_tables/ndvi_table_all_years_01.pkl")
ndvi_df$scl_class <- factor(ndvi_df$scl_class, seq(2, 11))
ntest <- 50
set.seed(1234)
ndvi_df_sample <- ndvi_df[sample.int(nrow(ndvi_df), ntest), ]
ndvi_test_set <- list()
for (i in 1:ntest)
    ndvi_test_set[[i]] <- ndvi_df_sample[i, ]

## this function only goes two layer deep, better use `recursive` function below
# reduce_size_while_keeping_prediction <- function(current_obj) {
#     orginal_obj <- current_obj
#     if (object.size(current_obj) < 1e+5) {
#         return (current_obj)
#     }
#     if (length(current_obj) == 1) {
#         return (current_obj)
#     } else {
#         for (first_layer_entry in names(current_obj)) {
#             temp <- current_obj
#             temp_entry <- temp[[first_layer_entry]]
#             temp[[first_layer_entry]] <- NULL
#             y_hat <- predict(orginal_obj, ndvi_df_sample)
#             y_hat_new <- try(predict(temp, ndvi_df_sample), silent = TRUE)
#             if (!identical(y_hat, y_hat_new)) {
#                 temp[[first_layer_entry]] <- temp_entry
#             }
#             current_obj <- temp
#             if (is.list(current_obj[[first_layer_entry]])) {
#                 for (entry_name in names(current_obj)) {
#                     # print(entry_name)
#                     # name <- paste0(current_obj_name,"$",entry_name)
#                     temp <- current_obj
#                     temp_entry <- temp[[first_layer_entry]][[entry_name]]
#                     temp[[first_layer_entry]][[entry_name]] <- NULL
#                     y_hat <- predict(orginal_obj, ndvi_df_sample)
#                     y_hat_new <- try(predict(temp, ndvi_df_sample), silent = TRUE)
#                     if (!identical(y_hat, y_hat_new)) {
#                         temp[[first_layer_entry]][[entry_name]] <- temp_entry
#                     }
#                     current_obj <- temp
#                 }
#             }
#         }
#         return (current_obj)
#     }
# }


assign_to_path <- function(current_obj, path_within_obj, assignment) {
    current_entry <- path_within_obj[1]
    if (length(path_within_obj) == 1) {
        current_obj[[current_entry]] <- assignment
        return (current_obj)
    }
    path_within_obj <- path_within_obj[-1]
    current_obj[[current_entry]] <- assign_to_path(current_obj[[current_entry]], path_within_obj, assignment)
    current_obj
}
is_object_equivalent <- function(obj, obj_untouched) {
    identical(predict(obj_untouched, ndvi_df_sample), try(predict(obj, ndvi_df_sample), silent = TRUE))
}
recursive  <- function(obj, path_within_obj = c(), obj_current = NULL, obj_untouched = NULL) {
    if (verbose) {
        cat(".")
    }
    if (is.null(obj_untouched))
        obj_untouched <- obj
    if (is.null(obj_current))
        obj_current <- obj
    if (is.list(obj) && !is.data.frame(obj)) {
        for (list_entry in names(obj)) {
            if (list_entry %in% c("forest")) {
                obj[[list_entry]] <- obj[[list_entry]]
            } else {
                obj[[list_entry]] <- recursive(obj[[list_entry]], c(path_within_obj, list_entry),
                    obj_current = obj_current, obj_untouched = obj_untouched)
            }
        }
    } else {
        temp_entry <- obj # extract_path_from_obj(obj, path_within_obj)
        obj_current <- assign_to_path(obj_current, path_within_obj, NULL)
        if (is_object_equivalent(obj_current, obj_untouched)) {
            # obj <- assign_to_path(obj, path_within_obj, temp_entry)
            return (NULL)
        } else {
            return (temp_entry)
        }
    }
    obj
}


verbose <- TRUE

files <- as.list(list.files("./data/computation_results/ml_models/R", full.names = TRUE))

reduce_one_obj <- function(f_old) {
    f_new <- gsub("/R/", "/R_small/", f_old)
    old_obj <- readRDS(f_old)
    y_hat <- predict(old_obj, ndvi_df_sample)
    new_obj <- recursive(old_obj)
    stopifnot(identical(y_hat, predict(new_obj, ndvi_df_sample)))
    if (verbose) {
        cat("\n", f_new, "\nlength: ", length(old_obj), "vs", length(new_obj),
            "\nsize: ", object.size(old_obj), "vs", object.size(new_obj), "\n",
            "reduced size by ", 100 - round(object.size(new_obj) / object.size(old_obj) * 100, 4), "% \n")
    }
    file.remove(f_new)
    saveRDS(duplicate(new_obj), f_new, refhook = function(x) "")

    reloaded_new <- readRDS(f_new, refhook = function(x) NULL)
    stopifnot(identical(y_hat, predict(reloaded_new, ndvi_df_sample)))
    if (identical(old_obj, reloaded_new)) {
        cat("nothing has changed in ", f_new, "\n")
    }
    if (object.size(new_obj) != object.size(reloaded_new))
        print("object sizes non-equal")
    rm(reloaded_new)
    rm(old_obj)
    rm(new_obj)
    gc()
}

for (f_old in files) {
    reduce_one_obj(f_old)
}

# this eats too much RAM:
# mclapply(files, reduce_one_obj, mc.cores = max(1, detectCores() - 3))
