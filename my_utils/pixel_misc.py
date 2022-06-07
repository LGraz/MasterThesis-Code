import pandas as pd
import numpy as np
import copy

import my_utils.strategies as strategies


def get_pixel_info_df(pix, itpl_methods_dict):
    """
    this function produces a data frame for a pixel with the following info:
    1. pixel.cov
    2. ndvi observed & interpolated
    """
    # add ndvi-observed
    ndvi_observed = pd.DataFrame({"ndvi_observed": pix.get_ndvi()})

    is_scl45 = (4 == pix.cov.scl_class.to_numpy()) | (
        5 == pix.cov.scl_class.to_numpy())
    ind_scl45 = np.where(is_scl45)
    pix._prepare_itpl("_")
    # get ndvi-itpl
    for itpl_args, itpl_kwargs in copy.deepcopy(itpl_methods_dict):
        # fit with only using scl45
        pix.itpl(*itpl_args, **itpl_kwargs)

        # out-of-bag estimates for scl45
        name = itpl_args[0]
        itpl_args = itpl_args[1:]
        itpl_kwargs_copy = itpl_kwargs.copy()
        itpl_strategy = itpl_kwargs_copy.pop("itpl_strategy")
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
    for label in drop_labels:
        big_df.drop(labels=label, axis=1, inplace=True)
    return big_df
