#%%
import os
import sys
import matplotlib.pyplot as plt
while "code/interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.itpl
import my_utils.strategies

from my_utils.data_handle import get_pixels
pixels = get_pixels(0.001, cloudy=True, train_test="train", seed=4321)
pix = pixels[13]
pix.plot_ndvi(colors="scl45")


for times in range(3):
	pix.itpl("Iteration" + str(times), my_utils.itpl.smoothing_spline,
             my_utils.strategies.robust_reweighting, smooth="gdd", times=times)
	pix.plot_itpl_df("Iteration" + str(times), label=str(times))
plt.legend()
plt.savefig('../latex/figures/interpol/robustifing_example.pdf',
            bbox_inches='tight')
