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
# import my_utils.loess as loess
import my_utils.itpl as itpl
importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel

np.random.seed(123)
pixels = data_handle.get_pixels(0.0008, seed=None, cloudy=True)
mein_obj = pixels[0]
# %%

print(type(itpl.loess))
mein_obj.plot_ndvi(scl_color=True)
mein_obj.itpl("loess", itpl.loess, alpha=0.2)
mein_obj.itpl("manual_loess", itpl.manual_loess, alpha=0.2)
mein_obj.plot_itpl_df("loess")
mein_obj.plot_itpl_df("manual_loess")
plt.legend()
# %%

y = mein_obj.get_ndvi()
x = mein_obj.cov.gdd.to_numpy()
xx = mein_obj.itpl_df.gdd.to_numpy()
weights = np.ones(len(x))
weights[14] = 0
weights[12] = 0
print(weights)

result = itpl.my_utils_loess(
    x, y, xx=xx, robust=False, weights=weights, alpha=0.4, deg=2)
mein_obj.plot_ndvi()
plt.plot(xx, result)

# %%
