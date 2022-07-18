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


name = "2017-20"
# name = "all_years"

###################################################
# plot stepwise
###################################################
from my_utils.plot import plot_ndvi_corr_step
from my_utils.data_handle import get_pixels
pixels = get_pixels(0.001, cloudy=True, train_test="train", seed=4321)
corr_method_name = "rf"
response = "ndvi_itpl_ss_noex_rob_rew_1"

plot_ndvi_corr_step(pixels[13], name, corr_method_name,
                    response, refit_before_rob=False)
# %%
