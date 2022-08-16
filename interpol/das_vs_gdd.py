#%%
import os
import sys
import matplotlib.pyplot as plt

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

import my_utils.plot_settings
from my_utils.plot import plot_ndvi_corr_step
from my_utils.data_handle import get_pixels

pixels = get_pixels(0.001, cloudy=True, train_test="train", seed=4321)

#%%
pix = pixels[13]
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[7, 2.5])
# fig.suptitle('Horizontally stacked subplots')
plt.sca(ax1)
pix.x_axis = "das"
pix.plot_ndvi(colors="scl45_grey", s=20)
plt.sca(ax2)
pix.x_axis = "gdd"
pix.plot_ndvi(colors="scl45_grey", s=20)

plt.savefig("../latex/figures/interpol/das_vs_gdd.pdf", bbox_inches="tight")
