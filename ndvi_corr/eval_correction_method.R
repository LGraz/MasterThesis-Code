require("reticulate") # to load pyhton data frame

# load data
path <- "data/computation_results/ndvi_tables/ndvi_table_all_years_001.pkl"
source_python("my_utils/R_read_pkl.py")
ndvi_df <- read_pickle_file("data/computation_results/ndvi_tables/ndvi_table_all_years_001.pkl")
ndvi_df$scl_class <- factor(ndvi_df$scl_class, seq(2, 11))
if (verbose) {
    str(ndvi_df)
}
