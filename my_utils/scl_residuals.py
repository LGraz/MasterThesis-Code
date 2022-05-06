import os
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
while "interpolation" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.data_handle as data_handle
from my_utils.pixel_multiprocess import pixel_multiprocess

# def _calc_res_help(pixels, scl_class, interpol_method, interpol_args):
#     method = "scl_45"
#     result = []
#     for pix in pixels:
#         ind = pix.filter(method)
#         getattr(pix, interpol_method)(**interpol_args)
#         pix.get_smoothing_spline(name=method, ind_keep=ind, smooth=0.3)
#         ind = pix.filter("scl"+str(scl_class))
#         for i, is_class in enumerate(ind):
#             if is_class:
#                 # `das` can be used to find the corresponding row in pix.step_interpolate
#                 das = pix.cov.das.iloc[i]
#                 das0 = pix.cov.das.iloc[0]
#                 obs = pix.get_ndvi().iloc[i]
#                 est = pix.step_interpolate[method].iloc[das-das0]
#                 temp = {"das":das, "i":i, "coord_id":pix.coord_id, 
#                 "year":pix.year, "scl_class":scl_class, "obs":obs, "est":est}
#                 result.append(temp)
#     return result

# def calc_residuals(pixels, scl_class, interpol_method, interpol_args):
#     """
#     descritpion
#     -----------
#     for every pixel in `pixels` we estimate the ndvi with `interpol_method` and cloud-free data (scl_class
#     in [4,5]) and compare it to the observed ndvi at 
#     times which are lables with `scl_class`
    
#     returns
#     -------
#     two lists: (ndvi_observed, ndvi_estimated),
#     where the i-th element of one corresponds 
#     to the i-th element of the other.  
#     """
#     final_result = []
#     n_cores = int(np.floor(os.cpu_count()*0.75)) # number of cores used
#     n = np.max([np.floor(len(pixels)/(10*n_cores)), 1])
#     # multiprocessing
#     res_list = []
#     pixels_partition = [pixels[i:i+n] for i in range(0, len(pixels), n)]    
#     args={"scl_class":scl_class, 
#         "interpol_method": interpol_method, 
#         "interpol_args": interpol_args}
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
#         res_list = [executor.submit(
#             _calc_res_help, pixels_chunk, scl_class, interpol_method, interpol_args) 
#             for pixels_chunk in tqdm(pixels_partition)]
#     # get result
#     for res in res_list:
#         final_result.extend(res.result())
#     return pd.DataFrame(final_result)
####################################


def _calc_res_help(pix, scl_class, interpol_method, **interpol_args):
    method = "scl_45"
    result = []
    ind = pix.filter(method)
    getattr(pix, interpol_method)(**{**interpol_args, "name":"scl_45"})
    ind = pix.filter("scl"+str(scl_class))
    for i, is_class in enumerate(ind):
        if is_class:
            # `das` can be used to find the corresponding row in pix.step_interpolate
            das = pix.cov.das.iloc[i]
            das0 = pix.cov.das.iloc[0]
            obs = pix.get_ndvi().iloc[i]
            est = pix.step_interpolate[method].iloc[das-das0]
            temp = {"das":das, "i":i, "coord_id":pix.coord_id, 
            "year":pix.year, "scl_class":scl_class, "obs":obs, "est":est}
            result.append(temp)
    return result
    
def calc_residuals(pixels, *args, **kwargs):
    result = pixel_multiprocess(pixels, _calc_res_help, *args, **kwargs)
    return result

def get_residuals(frac, scl_class, interpol_method, interpol_args, 
    interpol_method_args_str ,years = None, seed=4321, WW_cereals="cereals", save=True):
    """
    Description
    -----------
    this file is a wrapper to get observed and estimated ndvi

    Parameters:
    -----------
    frac, year, seed=4321, WW_cereals="WW" : args for data_handle.get_pixles()
    scl_class : desired scl_class for observations
    interpol_method, interpol_args : getattr(pix,"interpol_method")(interpol_args)
    interpol_method_args_str : string which identifies file-name
    """
    # first construct filename
    if WW_cereals=="WW":
        species = "_WW_"
    elif WW_cereals=="cereals":
        species= "_"
    else:
        Exception("WW_cereals must me 'WW' or 'cereals'")
    if years is None:
        years = [2017, 2018, 2019, 2020, 2021]
    if years == [2017, 2018, 2019, 2020, 2021]:
        year_str = "all_years"
    else:
        string = ""
        year_str = [string.append(str(year)[2:]) for year in years]
    print("caution!: Only WinterWheat assumed, to change in file name")
    file_name = "scl_"+species+interpol_method_args_str+"_"+str(scl_class)+\
        "_"+str(frac).replace(".","")+year_str+"_"+str(seed)
    file_path = "data/computation_results/scl/" + file_name + ".pkl"
    
    # second try load object, or generate it if fail 
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            result_df = pickle.load(f)
    else:
        pixels = data_handle.get_pixels(frac,cloudy=True,WW_cereals=WW_cereals, years=years, seed=seed)
        result_df = calc_residuals(
            pixels, scl_class, interpol_method, **interpol_args)
        if save:
            with open(file_path, "wb") as f:
                pickle.dump(result_df, f)
    return result_df


def plot_scl_class_residuals(scl_res_df, **kwargs):
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
    plt.axis('square')
    scl_class = scl_res_df.scl_class[0]
    plt.title(f"SCL class {scl_class}: {scl_description[scl_class]}")
    plt.scatter(scl_res_df.obs,scl_res_df.est, **kwargs)
    plt.xlim([0,1])
    plt.xlabel("observed ndvi")
    plt.ylim([0,1])
    plt.ylabel("estimated ndvi (with scl in [4,5])")
    plt.plot((0,1),(0,1), c="black", alpha=0.4)
