"""
this file has been replaced by iterpol/methods/cv/cv_itpl_res.py
"""

# # %%
# import os
# import sys
# import importlib
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# from tqdm import tqdm

# while "interpol" in os.getcwd():
#     os.chdir("..")
# sys.path.append(os.getcwd())
# import my_utils.pixel as pixel
# import my_utils.data_handle as data_handle
# import my_utils.cv as cv

# importlib.reload(data_handle)  # get changes in my_utils.pixel
# importlib.reload(pixel)  # get changes in my_utils.pixel
# importlib.reload(cv)  # get changes in my_utils.pixel
# #%%

# frac = 0.1
# pixels = data_handle.get_pixels(frac, seed=54321)


# # get some sequence from 0 to 1 with high resolution at the borders
# n = 4
# s = 0.005
# print("NOW: Smoothing splines: --------------------")
# ss_xx = [np.arctan(np.sinh(x)**3) / np.pi + 0.5
#          for x in np.arange(-n, n, step=s)]
# ss_file_name = "ss_res_list_" + \
#     str(frac).replace(".", "") + "_" + str(n) + "_" + str(s).replace(".", "")
# ss_file_path = "data/computation_results/" + ss_file_name + ".pkl"
# if os.path.isfile(ss_file_path):
#     print("load data")
#     with open(ss_file_path, "rb") as f:
#         ss_res_list = pickle.load(f)
#         print("data loaded")
# else:
#     ss_args = ("smooth", {"method": "get_smoothing_spline", "k": np.inf}, {})
#     ss_res_list = [cv.get_res_list(
#         pixels, x, *ss_args) for x in tqdm(ss_xx)]
#     with open(ss_file_path, "wb") as f:
#         pickle.dump(ss_res_list, f)


# # %%
# ss_med = [cv.fun_to_optimize(res_list=x, statistic=np.median)
#           for x in ss_res_list]
# ss_075 = [cv.fun_to_optimize(res_list=x, statistic=cv.quantile(0.75))
#           for x in ss_res_list]
# ss_mse = [cv.fun_to_optimize(res_list=x, statistic=cv.mse)
#           for x in ss_res_list]


# plt.plot(ss_xx, ss_med)
# plt.plot(ss_xx, ss_075)
# plt.plot(ss_xx, ss_mse)

# # %%
# # plot quantile(residuals, q) for different q's
# for q in np.arange(0.4, 0.9, step=0.05):
#     obj = [cv.fun_to_optimize(res_list=x, statistic=cv.quantile(q))
#            for x in ss_res_list]
#     plt.plot(ss_xx, obj)
