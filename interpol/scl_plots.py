#%%
import os

import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel
import my_utils.data_handle as data_handle
import my_utils.cv as cv
import my_utils.scl_residuals as scl_res

importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel
importlib.reload(scl_res)

for class_nr in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
# for class_nr in [6]:
    temp_df = scl_res.get_residuals(1, class_nr, "get_smoothing_spline", 
        {"smooth":0.3},"ss03", seed=4321, WW_cereals="cereals", save=True)
    # scl_res.plot_scl_class_residuals(temp_df, alpha=0.1)
temp_df
# %%
