# %%
import os

import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import itertools

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

pixels = data_handle.get_pixels(0.01, seed=4321)

with open(f'./data/computation_results/kriging_med_param.pkl', 'rb') as f:
    kriging_med_param = pickle.load(f)

method_label_args = [
    # , ("get_cubic_spline", "cubic_spline", {})
    # , ("get_savitzky_golay", "savitzky_golay", {})
    ("get_smoothing_spline", "smoothing_spline", {"smooth": 0.01}), ("get_b_spline", "b_spline", {"smooth": 1}), ("get_ordinary_kriging", "ordinary_kriging", {"ok_args": {"variogram_model": "gaussian", "variogram_parameters": list(kriging_med_param)}}), ("get_fourier", "fourier", {"opt_param": {"p0": [350, 1, 1, 1, 1, 1],
                                                                                                                                                                                                                                                                                                        "bounds": ([50, -1, -5, -5, -5, -5], [500, 2, 5, 5, 5, 5])}}), ("get_double_logistic", "double_logistic", {"opt_param": {"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
                                                                                                                                                                                                                                                                                                                                                                                                                                 "bounds": ([0, 0, 0, 10, 0, -1], [1, 1, 300, 300, 1, 0])}})
    # , ("get_whittaker", {})
    # , ("get_loess", {})
]
random.seed(4321)
pixels_chosen = random.sample(pixels, 30)
pixels_chosen = [pixels_chosen[i] for i in [2, 1, 7, 8, 9, 10, 12, 13, 14]]
ax_inds = itertools.product([0, 1, 2], [0, 1, 2])
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 10))

# plt.suptitle("Different NDVI-Series Examples")
for pix, ax_ind in zip(pixels_chosen, ax_inds):
    pix.x_axis = "das"
    plt.sca(ax[ax_ind[0], ax_ind[1]])
    pix.plot_ndvi()
    for method, label, args in method_label_args:
        args = {"name": label, **args}
        getattr(pix, method)(**args)
        pix.plot_itpl_df(label, label=label)
plt.sca(ax[2, 2])
plt.legend(loc="lower left", fontsize="large")
plt.savefig('../latex/figures/interpol/problem_illustration.pdf',
            bbox_inches='tight')


# %%
