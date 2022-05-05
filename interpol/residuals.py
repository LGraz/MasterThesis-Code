#%% 
from cmath import exp
import os
import sys
import importlib
import matplotlib
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
#%%
np.random.seed(4567)
pixels = data_handle.get_pixels(0.001, cloudy=True)

for pix in pixels:
    pix.get_smoothing_spline(smooth=0.1)

ind = pix.filter("scl_45")
pix.get_smoothing_spline(name="scl_45", ind_keep=ind, smooth=0.1)
pix.plot_ndvi("o")
pix.plot_step_interpolate(which="scl_45")
plt.show()

#%%
np.random.seed(4567)
interpol_args={"smooth":0.3, "name":"hi"}
pixels = data_handle.get_pixels(0.01, cloudy=True)
scl_classes = [1, 2, 3, 6, 7, 8, 9]
scl_class = 7


#%%
for class_nr in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    interpol_args={"smooth":0.3}
    obs, est = scl_res.get_residuals(0.1, class_nr, "get_smoothing_spline", interpol_args,"ss03")
    scl_res.plot_scl_class_residuals(obs, est, class_nr, alpha=0.1)


#%%
# EXAMPLE OF HIGH VALUE
# np.random.seed(4567)
# pixels = data_handle.get_pixels(0.01, cloudy=True)
# scl_classes = [1, 2, 3, 6, 7, 8, 9]
# scl_class = 7
# ndvi_obs = []
# ndvi_est = []
# method = "scl_45"
# failed = False
# for pix in pixels:
#     ind = pix.filter(method)
#     pix.get_smoothing_spline(name=method, ind_keep=ind, smooth=1)
#     ind = pix.filter("scl"+str(scl_class))
#     for i, is_class in enumerate(ind):
#         if is_class:
#             # `das` can be used to find the corresponding row in pix.step_interpolate
#             das = pix.cov.das.iloc[i]
#             das0 = pix.cov.das.iloc[0]
#             obs = pix.get_ndvi().iloc[i]
#             est = pix.step_interpolate[method].iloc[das-das0]
#             if est > 600:
#                 break
#             ndvi_obs.append(obs)
#             ndvi_est.append(est)
#     if failed:
#         temp = pix
#         break
# plt.figure()
# temp.plot_ndvi("o")
# temp.plot_step_interpolate(method)