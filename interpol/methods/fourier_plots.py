"""
plots a nice and a bad fourier and double-logistic plot
"""
# %%
import os
import sys
import numpy as np
import importlib
import matplotlib.pyplot as plt
while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.cv as cv
import my_utils.data_handle as data_handle
import my_utils.pixel as pixel
import my_utils.plot_settings


pixels = data_handle.get_pixels(0.005, seed=123)


def plot_fourier_and_doublelogistic(pix):
    pix.x_axis = "das"
    pix.get_fourier(opt_param={"p0": [350, 1, 1, 1, 1, 1],
                               "bounds": ([50, -1, -5, -5, -5, -5], [500, 2, 5, 5, 5, 5])})
    pix.plot_ndvi(ylim=[0.12, 1])
    pix.plot_itpl_df("fourier", label="2cd order fourier")
    pix.get_double_logistic(name="dl", opt_param={"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
                                                  "bounds": ([0, 0, 0, 10, 0, -1], [1, 1, 300, 300, 1, 0])})
    pix.plot_itpl_df("dl", label="double logistic")
    plt.legend(loc="lower center")


ratio = 0.55

plt.subplot(1, 2, 1)  # index 2
plt.title("Expected Behaviour")
plot_fourier_and_doublelogistic(pixels[62])
my_utils.plot_settings.set_plot_ratio(ratio)

plt.subplot(1, 2, 2)  # row 1, col 2 index 1
plt.title("Degenerated Example")
plot_fourier_and_doublelogistic(pixels[18])
my_utils.plot_settings.set_plot_ratio(ratio)


# %%
plt.savefig('../latex/figures/interpol/fourier_dl_comparison.pdf',
            bbox_inches='tight')
