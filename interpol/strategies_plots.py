# %%
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
from my_utils.cv import get_pix_cv_resiudals
import my_utils.data_handle as data_handle
import my_utils.strategies as strategies
import my_utils.itpl as itpl
from my_utils.pixel_multiprocess import pixel_multiprocess

pixels_frac = 0.005
pixels = data_handle.get_pixels(
    pixels_frac, seed=4321, cloudy=True, WW_cereals="cereals"
)
punish_factor = 1 / 2
optimization_param = {"p0": [0.2, 0.8, 300, 1700, 0.001, -0.001],
                      "bounds": ([0.1, 0, 0, 500, 0, -1], [1, 1, 1200, 5000, 1, 0])}

subset_ind = [1, 7, 9, 14, 15, 17, 18, 21, 41, 171, 172, 173]
enum_pixels_subset = [(i, pixels[i]) for i in range(len(pixels))
                      if i in subset_ind]
for i, pix in enum_pixels_subset:
    # pix = pixels[7]
    plt.title("nr " + str(i))
    pix.plot_ndvi(colors="scl45")
    for j in range(4):
        label = "dl_" + str(j)
        print(label)
        pix.itpl(label, itpl.double_logistic, strategies.robust_reweighting,
                 punish_negative=punish_factor, times=j,
                 opt_param=optimization_param, debug=False)
        pix.plot_itpl_df(label, label=label)
    plt.legend()
    plt.show()


# %%
pix.itpl(label, itpl.double_logistic, strategies.robust_reweighting,
         punish_negative=punish_factor, times=j,
         opt_param=optimization_param, debug=False)

# %%
pix = pixels[16]
plt.title("nr " + str(i))
pix.plot_ndvi(colors="scl45")
for j in range(4):
    label = "dl_" + str(j)
    pix.itpl(label, itpl.double_logistic, strategies.robust_reweighting, punish_negative=punish_factor, times=j,
             opt_param=optimization_param, debug=True)
    pix.plot_itpl_df(label, label=label)
plt.legend()
plt.show()

# %%
