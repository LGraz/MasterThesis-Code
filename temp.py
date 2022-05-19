# %%
import enum
import numpy as np
import pandas as pd
import os
import sys
import importlib
import matplotlib.pyplot as plt
from csaps import csaps

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.cv as cv
import my_utils.data_handle as data_handle
import my_utils.pixel as pixel
importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel

np.random.seed(123)
pixels = data_handle.get_pixels(0.0008, seed=None, cloudy=True)
self = pixels[0]
# %%


def smoothing_splines(x, y, xx, weights, *args, **kwargs):
    return csaps(x, y, xx, *args, weights=weights, **kwargs)


name = "bla"
itpl_strategy = identity
itpl_fun = smoothing_splines  # a function with (x, y, xx, weights)
filter_method_kwargs = ("filter_scl", {"classes": [4, 5]})

name, itpl_fun, itpl_strategy, filter_method_kwargs = [
    ("filter_scl", {"classes": [4, 5]})]


def itpl(self, name, itpl_fun, itpl_strategy, filter_method_kwargs=[("filter_scl", {"classes": [4, 5]})]):
    """
    parameters
    ----------
    name : string to save results in `self.itpl_df` 
    itpl_fun : a interpolation-function arguments (x, y, xx, weights)
    itpl_strategy : a function which applies `itpl_fun`
    filter_method_kwargs : a list of tupel("filter_name", {**filter_kwargs})
    """
    # prepare
    if name in self.itpl_df.columns:
        print("There already exists an collumn named: " + name)
    x, y, xx = self._prepare_itpl(name)

    # apply filter / weighting methods
    weights = np.asarray(([1] * len(x)))
    for filter_method, filter_kwargs in filter_method_kwargs:
        weights = filter_method(self, weights, **filter_kwargs)
        weights = getattr(self, filter_method)(weights, **filter_kwargs)

    # perform calcultions
    ind = np.where(weights > 0)
    result = itpl_strategy(
        itpl_fun, x[ind], y[ind], xx, weights[ind], smooth=0.2)

    # save result
    result = pd.DataFrame(result, columns=[name])
    if name in self.itpl_df.columns:
        self.itpl_df[name] = result.to_numpy()
    else:
        self.itpl_df = self.itpl_df.join(result)
    return result


# %%
