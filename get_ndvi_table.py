# %%
from copyreg import pickle
import os
import sys
import numpy as np
import pandas as pd
import pickle

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.data_handle as data_handle
import my_utils.itpl as itpl
import my_utils.strategies as strategies
from my_utils.pixel_multiprocess import pixel_multiprocess


def get_pixel_info_df(pix):
    """
    this function produces a data frame for a pixel with the following info:
    1. pixel.cov
    2. ndvi observed & interpolated
    """
    ndvi_observed = pd.DataFrame({"ndvi_observed": pix.get_ndvi()})

    name = "ndvi_itpl"
    pix.itpl(name, itpl.smoothing_spline, smooth=1e-6,
             itpl_strategy=strategies.identity_no_extrapol)

    itpl_df = pix.itpl_df[pix.itpl_df.is_observation].reset_index(
        drop=True)
    is_scl45 = (4 == pix.cov.scl_class.to_numpy()) | (
        5 == pix.cov.scl_class.to_numpy())
    ind = np.where(is_scl45)
    out_of_bag = pix.itpl(name + "_out_of_bag", itpl.smoothing_spline, strategies.cv,
                          cv_strategy=strategies.identity_no_extrapol, smooth=1e-6,
                          return_residuals=False)
    out_of_bag = out_of_bag[pix.itpl_df.is_observation.to_numpy()[0]]
    temp = itpl_df[name].to_numpy()
    temp[ind] = out_of_bag[~np.isnan(out_of_bag)]
    itpl_df[name] = temp

    big_df = pd.concat(
        [itpl_df, ndvi_observed, pix.cov.reset_index(drop=True)], axis=1)
    drop_labels = ['is_observation', 'coord_id', 'FID', 'scene_id', 'product_uri',
                   'x_coord', 'y_coord', 'epsg']
    for label in drop_labels:
        big_df.drop(labels=label, axis=1, inplace=True)
    return big_df


# %%
frac = 1
filename = f"data/computation_results/ndvi_itpl_VS_observed_and_rest{frac}.pkl"
pixels = data_handle.get_pixels(
    frac, cloudy=True, WW_cereals="cereals", seed=4321)


def help_fun(pix):
    df = get_pixel_info_df(pix)
    return [df]


if not os.path.exists(filename):
    print("generate: " + filename)
    df_list = pixel_multiprocess(pixels, help_fun)
    result = pd.concat(df_list, axis=0, ignore_index=True)
    with open(filename, "wb") as f:
        pickle.dump(result, f)
else:
    print(filename + " already generated")

# %%
