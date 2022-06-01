# %%
import os
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt


while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel
import my_utils.data_handle as data_handle
import my_utils.cv as cv
import my_utils.scl_residuals as scl_res

importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel
importlib.reload(scl_res)
# %%
np.random.seed(4567)
pixels = data_handle.get_pixels(0.001, cloudy=True)

for pix in pixels:
    pix.get_smoothing_spline(smooth=0.1)

pix.get_smoothing_spline(name="scl_45", smooth=0.1)
pix.plot_ndvi(colors="scl")
pix.plot_itpl_df(which="scl_45")
plt.savefig('../latex/figures/interpol/residuals_scl_classes.pdf',
            bbox_inches='tight')
