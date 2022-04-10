import numpy as np
import os
import concurrent.futures

import sys
while "interpolation" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel


# Statistics
def mse(array): return np.mean(np.square(array))


def quantile(q):
    return lambda array: np.quantile(np.abs(array), q)


def _muliprocess_help(pix, cv_args):
    return pix.cv_interpolation(**cv_args).to_numpy()


def get_res_list(pixel_list, param, param_name, cv_args, method_args):
    """
    Description
    -----------
    This function helps optimizing:
        statistic( residuals( interpol_method( param)))

    parameters
    ----------
    pixel_list : list of pixel-obj
    param : parameter used for interpolation method
    param_name : name of parameter as used in the interpolation method, eg: "smooth"
    cv_args: arguments for .cv_interpolation (excluding 'method_args')
    method_args : args passed from .cv_interpolation to method (excluding `param`)
    statistic : a meaningful function:  vector -> number,  like mean-square-error

    How-to-use
    ----------
    method_args = (
        "smooth", {"method": "get_smoothing_spline", "k":np.inf}, {}, quantile(0.75))
    # optimum calculated
    opt = scipy.optimize.minimize_scalar(fun_to_optimize, bounds=(0, 1),
                                   method="Bounded", args=ss_args_tpl_075)
    # optimal cross-validation value of statisitc:
    fun_to_optimize(opt["x"], *method_args)

    returns
    -------
    list with residuals (numpy arrays)    
    """
    cv_args = {**cv_args, "method_args": {**method_args, param_name: param}}
    res_list = []
    # multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        res_list = [executor.submit(
            _muliprocess_help, pix, cv_args=cv_args) for pix in pixel_list]
    # get result
    res_list = [res.result() for res in res_list]
    # trim first and last observation (and convert to numpy)
    res_list = [res[1:(len(res) - 1)] for res in res_list]
    return res_list


def fun_to_optimize(param, param_name, cv_args, method_args, pixel_list, statistic=quantile(0.5), res_list=None):
    if res_list is None:
        res_list = get_res_list(
            pixel_list, param, param_name, cv_args, method_args)
    return statistic([statistic(res) for res in res_list])
