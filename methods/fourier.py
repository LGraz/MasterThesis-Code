# %%
import os
import sys
import numpy as np
import importlib
import matplotlib.pyplot as plt
while "methods" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.cv as cv
import my_utils.data_handle as data_handle
import my_utils.pixel as pixel

# %%
importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel

np.random.seed(123)
pixels = data_handle.get_pixels(0.0008)
for pix in pixels:
    try:
        obj, popt = pix.get_fourier()
        pix.plot_ndvi()
        pix.plot_step_interpolate("fourier")
        plt.show()
    except:
        print(f"Failed to find optim parameters for pixel {pix.coord_id}")
# ==> did not find good parameters (period of one day)

# %%

# AGAIN, but with init guess and some bounds
np.random.seed(123)
pixels = data_handle.get_pixels(0.0008)
for pix in pixels:
    try:
        obj, popt = pix.get_fourier(opt_param={"p0": [350, 1, 1, 1, 1, 1],
                                               "bounds": ([50, -1, -5, -5, -5, -5], [500, 2, 5, 5, 5, 5])})
        pix.plot_ndvi()
        pix.plot_step_interpolate("fourier")
        plt.show()
    except:
        print(f"Failed to find optim parameters for pixel {pix.coord_id}")
# ==> did not find good parameters (period of one day)
