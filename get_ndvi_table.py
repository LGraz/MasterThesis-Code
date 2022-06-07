# %%
import os
import sys
import pandas as pd
import pickle

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.data_handle as data_handle
import my_utils.itpl as itpl
import my_utils.strategies as strategies
from my_utils.pixel_multiprocess import pixel_multiprocess
from my_utils.pixel_misc import get_pixel_info_df

# settings
update = False  # True = recalculate everything
save = False
x_axis = "gdd"
frac = 0.0002

# load Data
pixels_path = "data/computation_results/pixels_for_ndvi_table__" + \
    x_axis + str(frac).replace(".", "") + ".pkl"
if os.path.exists(pixels_path):
    with open(pixels_path, "rb") as f:
        pixels = pickle.load(f)
else:
    pixels = data_handle.get_pixels(
        frac, cloudy=True, WW_cereals="cereals", seed=4321)
    for pix in pixels:
        pix.x_axis = x_axis


# list with all interpolation options:
# (tupel: itpl_args, dict: itpl_kwargs)
itpl_methods_dict = [
    (
        ("ndvi_itpl_ss", itpl.smoothing_spline),
        {"smooth": x_axis, "update": update,
            "itpl_strategy": strategies.identity_no_extrapol},
    ),
    (
        ("ndvi_itpl_loess", itpl.loess),
        {"alpha": x_axis, "update": update,
            "itpl_strategy": strategies.identity_no_extrapol},
    ),
    (
        ("ndvi_itpl_dl", itpl.double_logistic),
        {"itpl_strategy": strategies.identity,
            "update": update, "opt_param": x_axis}
    )
]


# %%


def help_fun(pix):
    df = get_pixel_info_df(pix, itpl_methods_dict)
    return [df]


df_list = pixel_multiprocess(pixels, help_fun)
ndvi_table = pd.concat(df_list, axis=0, ignore_index=True)

# %%
# save computation results
if save:
    with open(pixels_path, "wb") as f:
        pickle.dump(pixels, f)


# ndvi_table_path = f"data/computation_results/ndvi_itpl_VS_observed_and_rest{frac}.pkl"
# if not os.path.exists(ndvi_table_path):
#     print("generate: " + ndvi_table_path)
#     df_list = pixel_multiprocess(pixels, help_fun)
#     result = pd.concat(df_list, axis=0, ignore_index=True)
#     with open(ndvi_table_path, "wb") as f:
#         pickle.dump(result, f)
# else:
#     print(ndvi_table_path + " already generated")


# %%
help_fun(pixels[0])
# %%
