# install.packages("reticulate")
path <- "data/computation_results/ndvi_tables/ndvi_table/ndvi-table_0.01"

require("reticulate")
print(getwd())
source_python("/home/lukas/Documents/ETH/MASTER_THESIS/code/my_utils/R_read_pkl.py")
pickle_data <- read_pickle_file(path)

str(pickle_data)
