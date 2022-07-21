# %%
import os
import sys
import numpy as np
import pandas as pd
import copy

while "ndvi" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

from my_utils import data_handle, pixel, itpl, strategies, pixel_multiprocess

frac = 0.01
data_handle_kwargs = {
    "frac": frac,
    "years": [2017, 2018, 2019, 2020, 2021],
    "cloudy": True,
    "WW_cereals": "cereals",
    "seed": 4321,
}
pixels = data_handle.get_pixels(**data_handle_kwargs)

responses = [
    "ndvi_itpl_ss_noex",
    # "ndvi_itpl_loess_noex",
    "ndvi_itpl_dl",
    "ndvi_itpl_ss_noex_rob_rew_1",
    # "ndvi_itpl_loess_noex_rob_rew_1",
    "ndvi_itpl_dl_rob_rew_1",
]
short_names = [
    "rf",
    "lm_scl",
    "lm_all",
    # "lm_step",
    "mars",
    "gam",
    "lasso",
]


def get_ndvi_ts_with_specific_correction(
    pix: pixel.Pixel,
    response,
    short_name,
    strat,
    itpl_method=None,
    w_method=lambda x: 1 / x,
):
    corrected, uncertain = pix.get_ndvi_corr(short_name=short_name, response=response)
    if strat == "rob":
        itpl_args = (itpl_method, strategies.robust_reweighting)
        itpl_kwargs = {
            "y": corrected,
            "times": 1,
            "fit_strategy": strategies.identity_no_xtpl,
        }
    elif strat == "id":
        itpl_args = (itpl_method, strategies.identity_no_xtpl)
        itpl_kwargs = {"y": corrected}
    else:
        raise Exception("______________")
    itpl_kwargs = (
        {**itpl_kwargs, "smooth": "gdd"}
        if (itpl_method == itpl.smoothing_spline)
        else {**itpl_kwargs, "opt_param": "gdd"}
    )
    return pix.itpl("ss_rob45", *itpl_args, w=w_method(uncertain), **itpl_kwargs)


def get_dict_array_ndvi_ts(pix: pixel.Pixel):
    try:
        itpl_methods = {"ss": itpl.smoothing_spline, "dl": itpl.double_logistic}
        itpl_strats = {
            "rob": strategies.robust_reweighting,
            "id": strategies.identity_no_xtpl,
        }

        # "Dict-Array" with:   strats x methods x corr_ml
        dict_corr_ml = {short_name: None for short_name in short_names}
        dict_methods = {key: copy.deepcopy(dict_corr_ml) for key in itpl_methods.keys()}
        dict_strats = {key: copy.deepcopy(dict_methods) for key in itpl_strats.keys()}
        DICTS = dict_strats

        # fill "Dict-Array"
        for itpl_method in itpl_methods.keys():
            # use also identity_no_xtpl for doublelogistic for simplicity
            for strat in itpl_strats.keys():
                for approach in [None, "correct_ndvi_then_itpl"]:
                    if approach is None:
                        short_name = "no_correction"
                        DICTS[strat][itpl_method][short_name] = pix.itpl(
                            short_name + itpl_method + strat,
                            itpl_methods[itpl_method],
                            itpl_strategy=itpl_strats[strat],
                            smooth="gdd",
                        )
                    else:  # use NDVI correction
                        temp = (
                            ["ndvi_itpl_ss_noex", "ndvi_itpl_ss_noex_rob_rew_1"]
                            if (itpl_method == "ss")
                            else ["ndvi_itpl_dl", "ndvi_itpl_dl_rob_rew_1"]
                        )
                        relevant_response = temp[1] if (strat == "rob") else temp[0]
                        for short_name in short_names:
                            DICTS[strat][itpl_method][
                                short_name
                            ] = get_ndvi_ts_with_specific_correction(
                                pix,
                                relevant_response,
                                short_name,
                                strat,
                                itpl_method=itpl_methods[itpl_method],
                            )
        DICTS["yield"] = pix.yie.dry_yield
        DICTS["yield_verbose"] = pix.yie
        DICTS["gdd"] = np.asarray(pix.itpl_df.gdd)
        return DICTS
    except:
        return None


# a trick to load all R-correction-models
get_dict_array_ndvi_ts(pixels[0])

# %%
##############################################################
# Get Correction
##############################################################
import concurrent.futures

n_cores = int(np.min([np.floor(os.cpu_count() * 0.9), os.cpu_count() - 2]))
fname = f"./data/computation_results/pixels_itpl_corr_dict_array/{len(pixels)}.pkl"
if not os.path.exists(fname):
    from multiprocessing import Pool
    import tqdm

    # with Pool(processes = n_cores) as p:
    #     DICTS = list(
    #         tqdm.tqdm(p.imap_unordered(get_dict_array_ndvi_ts, pixels), total=len(pixels))
    #     )
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        DICTS = list(
            tqdm.tqdm(executor.map(get_dict_array_ndvi_ts, pixels), total=len(pixels))
        )

    DICTS = [DICT for DICT in DICTS if DICT is not None]
    data_handle.save(DICTS, fname)


# %%
