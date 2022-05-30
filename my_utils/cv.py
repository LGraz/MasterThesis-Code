import numpy as np
import os
from my_utils.pixel_multiprocess import pixel_multiprocess

import sys
while "interpolation" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel
import my_utils.strategies as strategies


def get_pix_cv_resiudals(pix: pixel.Pixel, itpl_fun,
                         cv_strategy=strategies.identity_no_extrapol,
                         par_name=None, par_value=None, **kwargs):
    """
    utility function for pixel_multiprocess, meant for parameter tuning.
    returns: list of residuals for each pixel (by fitting with pixel)
    """
    # add parameter to arguments
    if (par_name is not None) and (par_value is not None):
        kwargs = {**kwargs, par_name: par_value}
    # get residuals
    res = pix.itpl("cv_residuals", itpl_fun, strategies.cv,
                   cv_strategy=cv_strategy, return_residuals=True, **kwargs)
    return res[~np.isnan(res)]  # remove nan's
