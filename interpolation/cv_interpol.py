# %%
import os
import sys
import importlib
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

while "interpolation" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel
import my_utils.data_handle as data_handle
import my_utils.cv as cv


# %%
importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel

frac = 0.0002
np.random.seed(54321)
pixels = data_handle.get_pixels(frac)
# %%

ss_075_args_tpl = (
    "smooth", {"method": "get_smoothing_spline", "k": np.inf}, {}, pixels, cv.quantile(0.75))
# ss_075_opt = scipy.optimize.minimize_scalar(cv.fun_to_optimize, bounds=(
# 0, 1), method="Bounded", args=ss_075_args_tpl)
# print(ss_075_opt)

ss_med_args_tpl = (
    "smooth", {"method": "get_smoothing_spline", "k": np.inf}, {}, pixels, cv.quantile(0.5))
# ss_med_opt = scipy.optimize.minimize_scalar(cv.fun_to_optimize, bounds=(
# 0, 1), method="Bounded", args=ss_med_args_tpl)
# print(ss_075_opt)

ss_mse_args_tpl = (
    "smooth", {"method": "get_smoothing_spline", "k": np.inf}, {}, pixels, cv.mse)
# ss_mse_opt = scipy.optimize.minimize_scalar(cv.fun_to_optimize, bounds=(
# 0, 1), method="Bounded", args=ss_mse_args_tpl)
# print(ss_075_opt)

# %%
# print(cv.fun_to_optimize(ss_075_opt["x"] - 0.01, *ss_075_args_tpl))

xx = (np.linspace(-1, 1, num=3) * 10)
xx = [np.arctan(x) / np.pi + 0.5 for x in xx]

y_med = [cv.fun_to_optimize(x, *ss_med_args_tpl) for x in xx]

# %%
plt.scatter(xx, y_med)
# %%
res_list = cv.get_res_list(
    pixels, 0.2, "smooth", {"method": "get_smoothing_spline", "k": np.inf}, {})
# %%
[plt.plot(res) for res in res_list]
res_list
