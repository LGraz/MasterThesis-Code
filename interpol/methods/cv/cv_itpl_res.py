"""
Description
-----------
Tune parameters for itpl-methods with crossvalidation
Idea: for parameter, itpl-method do:
          calucalte loocv-residuals and put them all in a list
          muliply negative residuals by 1/2  (to make them less severe)
          apply several statistics to residuals
          chose best parameter for each statistic
"""

# %%
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

while "code/interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
from my_utils.cv import get_pix_cv_resiudals
import my_utils.data_handle as data_handle
import my_utils.strategies as strategies
import my_utils.itpl as itpl
from my_utils.pixel_multiprocess import pixel_multiprocess

pixels_frac = 0.2
pixels = data_handle.get_pixels(
    pixels_frac, seed=4321, cloudy=True, WW_cereals="cereals"
)
optim_itpl_param = dict()


def get_cv_residuals_dict(parameters=None, par_name=None, itpl_method=None):
    residuals = dict()
    for parameter in tqdm(parameters):
        try:
            res = pixel_multiprocess(
                pixels,
                get_pix_cv_resiudals,
                itpl_method,
                cv_strategy=strategies.identity_no_xtpl,
                par_name=par_name,
                par_value=parameter,
            )
        except Exception as e:
            print("failed at " + itpl_method.__name__ + str(parameter))
            print(e)
            res = np.nan
        residuals[str(parameter)] = res
    return residuals


def minimize_over_dict(residuals_dict, statistic):
    def multiply_negative_res(res, factor=0.5):
        res = np.array(res)  # make sure res is numpy
        neg_ind = np.where(res < 0)
        try:
            res[neg_ind] = factor * res[neg_ind]
        except IndexError as error:
            return np.nan
        return res
    temp = {k: statistic(multiply_negative_res(v))
            for k, v in residuals_dict.items()}
    k_min, v_min = (None, np.inf)
    for k, v in temp.items():
        if v < v_min:
            k_min, v_min = (k, v)
    return k_min, v_min


def plot_pdf_cdf(axs, data, **kwargs):
    count, bins_count = np.histogram(data, bins=100, range=(-0.55, 0.55))
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    axs[0].plot(bins_count[1:], pdf, linewidth=0.4, **kwargs)
    axs[1].plot(bins_count[1:], cdf, linewidth=0.4, **kwargs)


args_dict_list = [
    {
        "par_name": "smooth",
        "itpl_method": itpl.smoothing_spline,
        "parameters": [2**i for i in np.linspace(-30, 0, num=80)],
    },
    {
        "par_name": "alpha",
        "itpl_method": itpl.loess,
        "parameters": [2**i for i in np.linspace(-2.5, -0.5, num=10)],
    },
    {
        "par_name": "smooth",
        "itpl_method": itpl.b_spline,
        "parameters": [2**i for i in np.linspace(-10, 1, num=40)],
    },
    # {"par_name": None, "itpl_method": None, "parameters": None},
]


statistic_dict = {
    "rmse": lambda res: np.sqrt(np.mean(np.square(res))),
    "quantile50": lambda res: np.quantile(np.abs(res), 0.50),
    "quantile75": lambda res: np.quantile(np.abs(res), 0.75),
    "quantile85": lambda res: np.quantile(np.abs(res), 0.85),
    "quantile90": lambda res: np.quantile(np.abs(res), 0.90),
    "quantile95": lambda res: np.quantile(np.abs(res), 0.95),
}

# %%
for mmm in ["gdd", "das"]:
    # set time-method correctly
    for pix in pixels:
        pix.x_axis = mmm

    for args_dict in args_dict_list:
        par_name = args_dict["par_name"]
        itpl_method = args_dict["itpl_method"]
        parameters = args_dict["parameters"]
        param_str = "param_" + itpl_method.__name__ + "__" + mmm + "_" + par_name
        print("get: " + param_str + " ---------------------------------")

        # get residuals
        file_name = (
            "cv_residuals_per_param_and_method__"
            + param_str
            + "__"
            + str(pixels_frac)
            + str(np.sum(parameters)).replace(".", "")
        )
        file_path = "data/computation_results/cv_itpl_res/" + file_name
        residuals_dict = data_handle.load(file_path)
        if residuals_dict is None:
            residuals_dict = get_cv_residuals_dict(**args_dict)
            data_handle.save(residuals_dict, file_path)

        # apply statistics
        for name_, statistic in statistic_dict.items():
            optim_itpl_param[param_str + "_" + name_] = minimize_over_dict(
                residuals_dict, statistic
            )

        # raise error if parameter optimization failed
        if (
            optim_itpl_param[param_str + "_" + name_][0]
            in [parameters[0], parameters[-1]]
        ) or (
            optim_itpl_param[param_str + "_" + name_][0]
            in [parameters[0], parameters[-1]]
        ):
            raise Exception(
                "optimized parameter on the edge of searched parameters, adapt search"
            )

        # plot cdf
        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(5)
        fig.set_figwidth(11)
        cmap = matplotlib.cm.get_cmap("Spectral")
        n_ = len(residuals_dict.keys())
        i = 0
        for param, residuals in residuals_dict.items():
            rgba = cmap(i / n_)
            i += 1
            plot_pdf_cdf(axs, residuals, label=str(
                np.around(float(param), 1)), c=rgba),
        # plt.legend(ncol=5)
        plt.suptitle(itpl_method.__name__)
        plt.show()
        # print(residuals_dict.keys())

# save results
data_handle.save(optim_itpl_param,
                 "data/computation_results/cv_itpl_res/optim_itpl_param")

print(optim_itpl_param)

# %%
with open("data/computation_results/cv_itpl_res/optim_itpl_param", "rb") as f:
    optim_itpl_param = pickle.load(f)
pix = pixels[27]
pix.itpl("test", itpl.smoothing_spline,
         strategies.identity_no_xtpl, smooth=pix.x_axis)
pix.plot_itpl_df("test")
pix.plot_ndvi(colors="scl45")
