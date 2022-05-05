import os
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
from tqdm import tqdm
while "interpolation" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel
import my_utils.data_handle as data_handle

def calc_residuals(pixels, scl_class, interpol_method, interpol_args):
    """
    descritpion
    -----------
    for every pixel in `pixels` we estimate the ndvi with `interpol_method` and cloud-free data (scl_class
    in [4,5]) and compare it to the observed ndvi at 
    times which are lables with `scl_class`
    
    returns
    -------
    two lists: (ndvi_observed, ndvi_estimated),
    where the i-th element of one corresponds 
    to the i-th element of the other.  
    """
    ndvi_obs = []
    ndvi_est = []
    method = "scl_45"
    failed = False
    for pix in pixels:
        ind = pix.filter(method)
        getattr(pix, interpol_method)(**interpol_args)
        pix.get_smoothing_spline(name=method, ind_keep=ind, smooth=0.3)
        ind = pix.filter("scl"+str(scl_class))
        for i, is_class in enumerate(ind):
            if is_class:
                # `das` can )be used to find the corresponding row in pix.step_interpolate
                das = pix.cov.das.iloc[i]
                das0 = pix.cov.das.iloc[0]
                obs = pix.get_ndvi().iloc[i]
                est = pix.step_interpolate[method].iloc[das-das0]
                if est > 600:
                    break
                ndvi_obs.append(obs)
                ndvi_est.append(est)
        if failed:
            temp = pix
            break
    return (ndvi_obs, ndvi_est)


def get_residuals(frac, scl_class, interpol_method, interpol_args, interpol_method_args_str ,years = None, seed=4321, WW_cerals="WW"):
    """
    Description
    -----------
    this file is a wrapper to get observed and estimated ndvi

    Parameters:
    -----------
    frac, year, seed=4321, WW_cerals="WW" : args for data_handle.get_pixles()
    scl_class : desired scl_class for observations
    interpol_method, interpol_args : getattr(pix,"interpol_method")(interpol_args)
    interpol_method_args_str : string which identifies file-name
    """
    if years is None:
        years = [2017, 2018, 2019, 2020, 2021]
    if years == [2017, 2018, 2019, 2020, 2021]:
        year_str = "all_years"
    else:
        string = ""
        year_str = [string.append(str(year)[2:]) for year in years]
    print("caution!: Only WinterWheat assumed, to change in file name")
    file_name = "scl_res_"+interpol_method_args_str+"_"+str(scl_class)+\
        "_"+str(frac).replace(".","")+year_str+"_"+str(seed)
    file_path = "data/computation_results/scl/" + file_name + ".pkl"
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            observed_and_estimated_ndvi = pickle.load(f)
    else:
        np.random.seed(seed)
        pixels = data_handle.get_pixels(frac,cloudy=True,WW_cereals=WW_cerals, years=years)
        observed_and_estimated_ndvi = calc_residuals(
            pixels, scl_class, interpol_method, interpol_args)
        with open(file_path, "wb") as f:
            pickle.dump(observed_and_estimated_ndvi, f)
    return observed_and_estimated_ndvi


def plot_scl_class_residuals(ndvi_obs, ndvi_est, scl_class, **kwargs):
    "plots:   ndvi_observed  *VS*  ndvi_estimated"
    scl_description ={
        0 : "No Data (Missing data on projected tiles)",
        1 : "Saturated or defective pixel",
        2 : "Dark features / Shadows",
        3 : "Cloud shadows",
        4 : "Vegetation",
        5 : "Bare soils / deserts",
        6 : "Water",
        7 : "Cloud low probability",
        8 : "Cloud medium probability",
        9 : "Cloud high probability",
        10 :  "Thin cirrus",
        11 :  "Snow or ice"
    }
    plt.figure()
    plt.title(f"SCL class {scl_class}: {scl_description[scl_class]}")
    plt.scatter(ndvi_obs,ndvi_est, **kwargs)
    plt.xlim([0,1])
    plt.xlabel("observed ndvi")
    plt.ylim([0,1])
    plt.ylabel("estimated ndvi (with scl in [4,5])")
    
