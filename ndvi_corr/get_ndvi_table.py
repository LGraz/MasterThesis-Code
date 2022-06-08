# %%
import os
import sys

while "ndvi" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

import my_utils.get_ndvi_table as get_ndvi_table

get_ndvi_table.get_ndvi_table(1, update=True)


# >>> some test setup: ##############
# # %%
# from my_utils.data_handle import get_pixels
# import my_utils.strategies as strategies
# import copy
# import my_utils.itpl as itpl
# pix = get_pixels(0.001, seed=4321)[0]
# pix.itpl("dl", itpl.double_logistic, opt_param="gdd")

# x_axis = "gdd"
# update = False
# save = True
# itpl_methods_dict = [
#     (
#         ("ndvi_itpl_ss_noex", itpl.smoothing_spline),
#         {"smooth": x_axis, "update": update,
#          "itpl_strategy": strategies.identity_no_extrapol},
#     ),
#     (
#         ("ndvi_itpl_loess_noex", itpl.loess),
#         {"alpha": x_axis, "update": update,
#          "itpl_strategy": strategies.identity_no_extrapol},
#     ),
#     (
#         ("ndvi_itpl_dl", itpl.double_logistic),
#         {"itpl_strategy": strategies.identity,
#          "update": update, "opt_param": x_axis}
#     )
# ]
# # add robus_reweighting method for each method above
# for itpl_method_i in copy.deepcopy(itpl_methods_dict):
#     for j in [1]:  # j = times
#         temp_args, temp_kwargs = itpl_method_i
#         temp_kwargs["itpl_strategy"] = strategies.robust_reweighting
#         temp_kwargs["multiply_negative_res"] = 2
#         temp_kwargs["times"] = j
#         # name:
#         temp_args = list(temp_args)
#         temp_args[0] = temp_args[0] + "_rob_rew_" + str(j)
#         temp_args = tuple(temp_args)
#         # add to our methods
#         itpl_methods_dict.append((temp_args, temp_kwargs))
# get_ndvi_table.get_pixel_info_df(pix, itpl_methods_dict=itpl_methods_dict)
