"""plot smoothingspliens fit"""
# %%
import os
import sys
import matplotlib.pyplot as plt
while "ndvi" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.itpl
import my_utils.strategies

from my_utils.data_handle import get_pixels
pixels = get_pixels(0.001, cloudy=True, train_test="train", seed=4321)
pix = pixels[13]
pix.plot_ndvi(colors="scl45")
for q in ["50", "75", "85", "90", "95"]:
    pix.itpl("ss" + q, my_utils.itpl.smoothing_spline,
             my_utils.strategies.identity, smooth="gdd" + q)
    pix.plot_itpl_df("ss" + q)
plt.title("Smoothing Splines")
plt.savefig('../latex/figures/interpol/simple_example_ss.pdf',
            bbox_inches='tight')


# %%
pix.plot_ndvi(colors="scl45")
pix.itpl("loess75", my_utils.itpl.loess,
         my_utils.strategies.identity, alpha="gdd" + "75")
pix.plot_itpl_df("loess75")
plt.title("LOESS")
plt.savefig('../latex/figures/interpol/simple_example_loess.pdf',
            bbox_inches='tight')

# %%
