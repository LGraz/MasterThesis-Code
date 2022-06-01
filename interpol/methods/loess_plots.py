# %%
import enum
from re import A
import numpy as np
import pandas as pd
import os
import sys
import importlib
import matplotlib.pyplot as plt
from csaps import csaps
import time

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.cv as cv
import my_utils.data_handle as data_handle
import my_utils.pixel as pixel
import my_utils.strategies as strategies
from sklearn.model_selection import KFold
# import my_utils.loess as loess
import my_utils.itpl as itpl
importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(strategies)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel

np.random.seed(123)
pixels = data_handle.get_pixels(0.0008, seed=None, cloudy=True)
pix = pixels[0]
# %%
# loess is not "smooth" by nature:
pix.plot_ndvi(colors="scl")
pix.itpl("loess", itpl.loess, alpha=0.5)
# pix.itpl(, itpl.manual_loess, alpha=0.2)
pix.plot_itpl_df("loess")
# pix.plot_itpl_df("manual_loess")
plt.legend()
# %%


pix.plot_ndvi(colors="scl")
for i in [0, 1, 2, 3]:
    name = "robustLOESS" + str(i)
    pix.itpl(name, itpl.loess,
             itpl_strategy=strategies.robust_reweighting, deg=2,
             times=i, alpha=0.6, punish_negative=2)
    pix.plot_itpl_df(name)
plt.legend()
plt.show()
# %%
pix.plot_ndvi(colors="scl45")
for i in [0, 1, 2, 3, 4, 5, 6, 7]:
    name = "SmoothingSplines" + str(i)
    pix.itpl(name, itpl.smoothing_spline,
             itpl_strategy=strategies.robust_reweighting, times=i,
             smooth=0.000005, punish_negative=4, debug=True)
    pix.plot_itpl_df(name)
    # time.sleep(1)
plt.legend()
plt.show()

# %%
from csaps import csaps
x = pix.cov.gdd.to_numpy()
y = pix.get_ndvi()
xx = pix.itpl_df.gdd.to_numpy()
weights = np.ones(len(x))
# weights[12] = 0.0001
ypred = csaps(x, y, xx, weights=weights, smooth=0.000005)
pix.plot_ndvi(colors="scl")
plt.plot(xx, ypred)

# %%
# inspect smoothing spline divede 0
pix.itpl(name, itpl.smoothing_spline,
         itpl_strategy=strategies.robust_reweighting, times=i,
         smooth=0.2, filter_method_kwargs=[])

# %%
# CV - testing

pix.itpl("residuals", itpl.smoothing_spline,
         strategies.cv, smooth=0.000005)
pix.plot_ndvi(colors="scl")
pix.plot_itpl_df("residuals")


# %%
