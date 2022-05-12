#%%
import os
import itertools
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

ax_inds = itertools.product([0,1,2],[0,1,2])
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15,15))
for class_nr, ax_ind in zip([2, 3, 4, 5, 6, 7, 8, 9, 10], ax_inds):
    plt.sca(ax[ax_ind[0],ax_ind[1]])
# for class_nr in [6]:
    temp_df = scl_res.get_residuals(1, class_nr, "get_smoothing_spline", 
        {"smooth":0.3},"ss03", seed=4321, WW_cereals="cereals", save=True)
    scl_res.plot_scl_class_residuals(temp_df, alpha=0.006)
print("figure generated, saving ...")
plt.savefig('../latex/figures/interpol/scl_residuals_scatter.png', dpi=200,
            bbox_inches='tight')
