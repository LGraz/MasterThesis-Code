# %%
import os
import sys
import numpy as np
import pandas as pd

while "ndvi" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

from my_utils import data_handle, pixel, itpl, strategies

frac = 0.01
data_handle_kwargs = {"frac": frac, "years": [
    2017, 2018, 2019, 2020, 2021], "cloudy": True, "WW_cereals": "cereals", "seed": 4321}
pixels = data_handle.get_pixels(**data_handle_kwargs)

responses = [
    "ndvi_itpl_ss_noex",
    # "ndvi_itpl_loess_noex",
    "ndvi_itpl_dl",
    "ndvi_itpl_ss_noex_rob_rew_1",
    # "ndvi_itpl_loess_noex_rob_rew_1",
    "ndvi_itpl_dl_rob_rew_1"
]
short_names = [
    "rf",
    "lm_scl",
    "lm_all",
    # "lm_step",
    "mars",
    "gam",
    "lasso"
]


def bla(pix: pixel.Pixel, response, short_name, strat, itpl_method=None, w_method=lambda x: 1 / x):
    corrected, uncertain = pix.get_ndvi_corr(
        short_name=short_name, response=response)
    if strat == "rob":
        itpl_args = (itpl_method, strategies.robust_reweighting)
        itpl_kwargs = {"y": corrected, "times": 1,
                       "fit_strategy": strategies.identity_no_xtpl}
    elif strat == "id":
        itpl_args = (itpl_method, strategies.identity_no_xtpl)
        itpl_kwargs = {"y": corrected}
    else:
        raise Exception("______________")
    itpl_kwargs = {**itpl_kwargs, "smooth": "gdd"} if (
        itpl_method == itpl.smoothing_spline) else {**itpl_kwargs, "opt_param": "gdd"}
    return pix.itpl("ss_rob45", *itpl_args,
                    w=w_method(uncertain), **itpl_kwargs)


pix = pixels[0]


itpl_methods = {"ss": itpl.smoothing_spline, "dl": itpl.double_logistic}
itpl_strats = {"rob": strategies.robust_reweighting,
               "id": strategies.identity_no_xtpl}

for itpl_method in itpl_methods.keys():
    # use also identity_no_xtpl for doublelogistic for simplicity
    for strat in itpl_strats.keys():
        for approach in [None, "correct_ndvi_then_itpl"]:
            if approach is None:
                print("iterpolate with strat")
            else:
                temp = ["ndvi_itpl_ss_noex", "ndvi_itpl_ss_noex_rob_rew_1"
                        ] if (itpl_method == "ss") else [
                    "ndvi_itpl_dl", "ndvi_itpl_dl_rob_rew_1"]
                relevant_response = temp[1] if (strat == "rob") else temp[0]
                for short_name in short_names:
                    # print(itpl_methods[itpl_method].__name__,
                    #       itpl_strats[strat].__name__, relevant_response, short_name)
                    obj = bla(pix, relevant_response, short_name, strat,
                              itpl_method=itpl_methods[itpl_method])

# %%

##############################################################
# Get Correction
##############################################################


##############################################################
# Get Save data into nice format
##############################################################
