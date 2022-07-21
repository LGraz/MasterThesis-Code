# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

while "ndvi" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.itpl as itpl
import my_utils.data_handle as data_handle

np.random.seed(4567)
pixels = data_handle.get_pixels(0.001, cloudy=True)

for pix in pixels:
    pix.itpl("ss", itpl.smoothing_spline, smooth="gdd")

pix.itpl("scl_45", itpl.smoothing_spline, smooth="gdd")
pix.plot_ndvi(colors="scl")
pix.plot_itpl_df(which="scl_45")
plt.savefig("../latex/figures/ndvi_corr/residuals_scl_classes.pdf", bbox_inches="tight")

# %%
