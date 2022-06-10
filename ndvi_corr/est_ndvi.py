# %%
import os
import sys
import pickle
import copy
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# change directory to code_dir
while True:
    changed = False
    for dir in ["my_utils", "interpol", "ndvi_corr"]:
        if ("code/" + dir) in os.getcwd():
            os.chdir("..")
            changed = True
    if not changed:
        break
sys.path.append(os.getcwd())

import sklearn.linear_model
# my librarys
import my_utils.get_ndvi_table as get_ndvi_table
import my_utils.ml_models as ml_models
################# END SETUP ###############################


ndvi_table = get_ndvi_table.get_ndvi_table(0.01, update=False)
ndvi_table
# %%

response_names = [
    'ndvi_itpl_ss_noex',
    # 'ndvi_itpl_loess_noex',
    # 'ndvi_itpl_dl',
    'ndvi_itpl_ss_noex_rob_rew_1',
    # 'ndvi_itpl_loess_noex_rob_rew_1',
    # 'ndvi_itpl_dl_rob_rew_1',
]


covariate_names = [
    'ndvi_observed',
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12',
    'cum_rain',
    'avg_temp',
    'day_rain',
    'max_temp',
    'min_temp',
    # 'scl_class', # this is not a factor!
    # 'harvest_year'
    # add again "gdd" (also in `get_ndvi_table()` + update)
]

# ndvi_table = ndvi_table.loc[ndvi_table['scl_class'].isin([4, 5])]
for response_name in response_names:
    ols = ml_models.learn(sklearn.linear_model.LinearRegression,
                          ndvi_table, response_name, covariate_names)
    y_pred = ols.predict(ndvi_table[covariate_names])
    y_itpl = ndvi_table[response_name].to_numpy()
    point_size = 1
    alpha = 0.2
    plt.figure(figsize=(10, 10))
    plt.scatter(y_itpl, y_pred, s=point_size, alpha=alpha, c="blue")
    plt.title("interpolated  VS [predicted (blue), observed (red)]")
    y_obs = ndvi_table["ndvi_observed"].to_numpy()
    plt.scatter(y_itpl, y_obs, s=point_size, alpha=alpha, c="red")
    plt.xlabel(response_name)
    plt.show()

# %%
