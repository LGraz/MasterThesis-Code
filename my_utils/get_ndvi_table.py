import pickle
import pandas as pd
import numpy as np
import copy
import os

import my_utils.data_handle as data_handle
import my_utils.itpl as itpl
import my_utils.strategies as strategies
from my_utils.pixel_multiprocess import pixel_multiprocess


def help_fun(pix, itpl_methods_dict):
    "help-fun for multiprocessing"
    try:
        df_pix_tpl = get_pixel_info_df(copy.deepcopy(pix), itpl_methods_dict)
    except Exception as e:  # if the above fails, we at least dont want to lose our calculations
        print(e)
        print("pix_dataframe is set to 'None' and ignored")
        df_pix_tpl = (None, pix)
    return [df_pix_tpl]


def get_pixels_for_ndvi_table(frac, x_axis="gdd", update=False, save=True):
    # first: itpl-methods
    itpl_methods_dict = [
        (
            ("ndvi_itpl_ss_noex", itpl.smoothing_spline),
            {"smooth": x_axis, "update": update,
                "itpl_strategy": strategies.identity_no_extrapol},
        ),
        (
            ("ndvi_itpl_loess_noex", itpl.loess),
            {"alpha": x_axis, "update": update,
                "itpl_strategy": strategies.identity_no_extrapol},
        ),
        (
            ("ndvi_itpl_dl", itpl.double_logistic),
            {"itpl_strategy": strategies.identity,
                "update": update, "opt_param": x_axis}
        )
    ]
    # add robus_reweighting method for each method above
    for itpl_method_i in copy.deepcopy(itpl_methods_dict):
        for j in [1]:  # j = times
            temp_args, temp_kwargs = itpl_method_i
            temp_kwargs["itpl_strategy"] = strategies.robust_reweighting
            temp_kwargs["multiply_negative_res"] = 2
            temp_kwargs["times"] = j
            # name:
            temp_args = list(temp_args)
            temp_args[0] = temp_args[0] + "_rob_rew_" + str(j)
            temp_args = tuple(temp_args)
            # add to our methods
            itpl_methods_dict.append((temp_args, temp_kwargs))

    # now actual function
    """
    Here we calculate a big table of ndvi (out of bag) estimates using 
    scl=[4,5], and various covariates
    """

    # load Data
    pixels_path = "data/computation_results/pixels_for_ndvi_table__" + \
        x_axis + str(frac).replace(".", "") + ".pkl"
    if os.path.exists(pixels_path) and (not update):
        with open(pixels_path, "rb") as f:
            pixels = pickle.load(f)
            print(f"{len(pixels)} (partly) modified Pixels have been loaded -----")
    else:
        pixels = data_handle.get_pixels(
            frac, cloudy=True, WW_cereals="cereals", seed=4321)
        for pix in pixels:
            pix.x_axis = x_axis
    if (pixels is None) or (len(pixels) == 0):
        raise Exception(
            "Data not generated succesfully, you are in : " + os.getcwd())

    # apply `get_pixel_info_df` to each pixel and save results
    df_pix_tpl_list = pixel_multiprocess(pixels, help_fun, itpl_methods_dict)
    pixels = [df_pix_tpl_list[i][1] for i in range(len(df_pix_tpl_list))]

    # save computation results
    if save:
        with open(pixels_path, "wb") as f:
            pickle.dump(pixels, f)

    # get idea how many pixels failed
    fail_count = 0
    for i, pix in enumerate(pixels):
        if hasattr(pix, "itpl_df"):
            i += 1
    print(f"{fail_count/len(pixels)} % pixels have no 'itpl_df' ({fail_count}/{len(pixels)})")


def get_ndvi_table():
    # get ndvi_table
    ndvi_table_list = [df_pix_tpl_list[i][0]
                       for i in range(len(df_pix_tpl_list)) if df_pix_tpl_list[i][0] is not None]
    ndvi_table = pd.concat(ndvi_table_list,
                           axis=0, ignore_index=True)

    # return
    if return_pixels:
        return (ndvi_table, pixels)
    else:
        return ndvi_table


def get_pixel_info_df(pix, itpl_methods_dict):
    """
    this function produces a data frame for a pixel with the following info:
    1. pixel.cov
    2. ndvi observed & interpolated

    returns: (DataFrame, pixel)  -- pixel is saved later to compute things only once
    """
    # add ndvi-observed
    ndvi_observed = pd.DataFrame({"ndvi_observed": pix.get_ndvi()})

    is_scl45 = (4 == pix.cov.scl_class.to_numpy()) | (
        5 == pix.cov.scl_class.to_numpy())
    ind_scl45 = np.where(is_scl45)
    pix._prepare_itpl("_")

    # get ndvi-itpl
    drop_itpl_oob_labels = []
    for itpl_args, itpl_kwargs in copy.deepcopy(itpl_methods_dict):
        # fit with only using scl45
        pix.itpl(*itpl_args, **itpl_kwargs)

        # out-of-bag estimates for scl45
        name = itpl_args[0]
        itpl_args = itpl_args[1:]
        itpl_kwargs_copy = itpl_kwargs.copy()
        itpl_strategy = itpl_kwargs_copy.pop("itpl_strategy")
        drop_itpl_oob_labels.append(name + "_out_of_bag")
        out_of_bag = pix.itpl(
            name + "_out_of_bag",
            *itpl_args,
            itpl_strategy=strategies.cv,
            cv_strategy=itpl_strategy,
            return_residuals=False,
            **itpl_kwargs_copy,
        )
        out_of_bag = out_of_bag[pix.itpl_df.is_observation.to_numpy()[0]]
        itpl_df = pix.itpl_df[pix.itpl_df.is_observation].reset_index(
            drop=True)
        temp = itpl_df[name].to_numpy()
        temp[ind_scl45] = out_of_bag[
            ~np.isnan(out_of_bag)
        ]  # substitute with out_of_bag estim.
        itpl_df[name] = temp

    # concatenate dataframes with same number of rows
    big_df = pd.concat(
        [itpl_df, ndvi_observed, pix.cov.reset_index(drop=True)], axis=1)

    # remove collumns
    drop_labels = [
        "is_observation",
        "coord_id",
        "FID",
        "scene_id",
        "product_uri",
        "x_coord",
        "y_coord",
        "epsg",
    ]
    drop_labels.extend(drop_itpl_oob_labels)
    for label in drop_labels:
        big_df.drop(labels=label, axis=1, inplace=True)
    return (big_df, pix)
