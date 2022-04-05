# %%
import my_utils.pixel as pixel
import os
import pandas as pd
import numpy as np
import importlib

while "interpolation" in os.getcwd():
    os.chdir("..")

WW = True  # only consider 'Winter Wheat' otherwise consider Cereals
if WW:
    path_cov = os.path.join("data/yieldmapping_data", "WW_covariates_tot.csv")
    path_met = os.path.join("data/yieldmapping_data", "WW_meteo_tot.csv")
    path_yie = os.path.join("data/yieldmapping_data", "WW_yield_tot.csv")
else:
    path_cov = os.path.join("data/yieldmapping_data",
                            "Cereals_covariates_tot.csv")
    path_met = os.path.join("data/yieldmapping_data", "Cereals_meteo_tot.csv")
    path_yie = os.path.join("data/yieldmapping_data", "Cereals_yield_tot.csv")
d_cov = pd.read_csv(path_cov)
d_met = pd.read_csv(path_met)
d_yie = pd.read_csv(path_yie)

# %%
importlib.reload(pixel)  # get changes in my_utils.pixel
np.random.seed(1)
temp = pixel.Pixel(d_cov, d_met, d_yie)


temp.cv_interpolation(
    methodname="ss", method="get_smooting_spline", kwargs={'name': "WW"}, one_result_column=True)


# %%
