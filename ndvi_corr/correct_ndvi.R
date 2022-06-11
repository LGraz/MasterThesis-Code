# load data
require("reticulate")
path <- "data/computation_results/ndvi_tables/ndvi_table_0.01"
source_python("my_utils/R_read_pkl.py")
pickle_data <- read_pickle_file("data/computation_results/ndvi_tables/ndvi_table_0.01")
str(pickle_data)

response = c(
    
)
"ndvi_itpl_ss_noex",
"ndvi_itpl_loess_noex",
"ndvi_itpl_dl",
"ndvi_itpl_ss_noex_rob_rew_1",
"ndvi_itpl_loess_noex_rob_rew_1",
"ndvi_itpl_dl_rob_rew_1",
"ndvi_observed",
"B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12",
"cum_rain",
"avg_temp",
"day_rain",
"max_temp",
"min_temp",
"scl_class",
"harvest_year"




model_lm
