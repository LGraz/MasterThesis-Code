# %%
import os
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
from my_utils.itpl import smoothing_spline
import pandas as pd

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel
import my_utils.data_handle as data_handle
import my_utils.cv as cv
import my_utils.scl_residuals as scl_res
import my_utils.itpl as itpl
import my_utils.strategies as strategies

frac = 0.001
pixels = data_handle.get_pixels(
    frac, cloudy=True, WW_cereals="cereals", seed=4321)

# %%

for pix in pixels:
    pix.x_axis = "gdd"
    pix.get_ndvi()


def abc(Pixel: pix):
    ndvi_observed = pd.DataFrame({"ndvi_observed": pix.get_ndvi()})

    name = "smoothing_splines"
    pix.itpl(name, itpl.smoothing_spline, smooth=1e-6,
             itpl_strategy=strategies.identity_no_extrapol)
    itpl_df = pix.itpl_df[pix.itpl_df.is_observation].reset_index(
        drop=True).drop(labels="is_observation", axis=1)
    return pd.concat([itpl_df, ndvi_observed, pix.cov.reset_index(drop=True)], axis=1)


pix = pixels[0]
abc(pix)
# %%
