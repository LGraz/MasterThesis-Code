"""
    fit double-logistic for some pixels
    identify interesting subset
    observe optimization failure
"""
# %%
import os
import sys
import matplotlib.pyplot as plt

while "code/interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.data_handle as data_handle
import my_utils.strategies as strategies
import my_utils.itpl as itpl

pixels_frac = 0.005
pixels = data_handle.get_pixels(
    pixels_frac, seed=4321, cloudy=True, WW_cereals="cereals"
)
multiply_negative = 2
optimization_param = {
    "p0": [0.2, 0.8, 300, 1700, 0.01, -0.01],
    "bounds": ([0.1, 0, 0, 500, 0, -1], [1, 1, 1200, 5000, 1, 0])}

subset_ind = range(len(pixels))
subset_ind = [15, 17, 18, 172, 173]
enum_pixels_subset = [(i, pixels[i]) for i in range(len(pixels))
                      if i in subset_ind]
for i, pix in enum_pixels_subset:
    # pix = pixels[7]
    plt.title("nr " + str(i))
    pix.plot_ndvi(colors="scl45")
    for j in range(4):
        label = "dl_" + str(j)
        # print(label)
        pix.itpl(label, itpl.double_logistic, strategies.robust_reweighting,
                 multiply_negative_res=multiply_negative, times=j,
                 opt_param=optimization_param, debug=False)
        pix.plot_itpl_df(label, label=label)
    plt.legend()
    plt.show()
