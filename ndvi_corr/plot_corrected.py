"""
stepwise plot of ndvi correction and fitting
1. estimate truth (i.e. out-of-bag interpolated value)
2. train model on to predict 'true ndvi' on data like
    truth ~ observed_ndvi + B02-B12 + weather
3. refit on estimated data


Returns:
    plots: stepwise procedure
"""
# %%
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

while "ndvi" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

import my_utils.plot_settings
from my_utils.data_processing.add_pseudo_factor_columns import add_pseudo_factor_columns

###################################################
# Learn Correction Model
###################################################

# data : load and prepare  ------------------------------------------
if False:  # year - leave out data?
    name = "2017-20"
    ndvi_table = pd.read_pickle(
        "data/computation_results/ndvi_tables/ndvi_table_2017-20_001.pkl")
else:
    name = "all_years"
    ndvi_table = pd.read_pickle(
        "data/computation_results/ndvi_tables/ndvi_table_0.01")
print("----------------  Data loaded")
ndvi_table, factor_encoding_colnames = add_pseudo_factor_columns(
    ndvi_table, "scl_class")
covariates = [
    "ndvi_observed",
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    # "cum_rain",
    # "avg_temp",
    # "day_rain",
    # "max_temp",
    # "min_temp",
    *factor_encoding_colnames  # "scl_class"
]
response = "ndvi_itpl_ss_noex_rob_rew_1"
X = ndvi_table[covariates]


# fit model   --------------------------------------------------------
# first ndvi
np.random.seed(4321)
forest = RandomForestRegressor(n_estimators=50, n_jobs=os.cpu_count() - 2)
forest.fit(X, ndvi_table[response])
print("---------------- model ndvi trained")

# second residuals (of ndvi)
forest_residuals = RandomForestRegressor(
    n_estimators=50, n_jobs=os.cpu_count() - 2)
res = ndvi_table[response] - forest.predict(X)
forest_residuals.fit(X, np.abs(res))
print("---------------- model residuals trained")

# %%
###################################################
# plot stepwise
###################################################
from my_utils.plot import plot_ndvi_corr_step
from my_utils.data_handle import get_pixels
pixels = get_pixels(0.001, cloudy=True, train_test="train", seed=4321)


plot_ndvi_corr_step(pixels[13], name, forest,
                    forest_residuals, covariates, refit_before_rob=False)
# %%
