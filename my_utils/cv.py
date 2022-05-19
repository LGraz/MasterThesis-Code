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
    return pix.cv_itpl(**cv_args)["cv_res_"].to_numpy()


def get_res_list(pixel_list, param, param_name, cv_args, method_args):
    """
    Description
    -----------
    This function helps optimizing:
        statistic( residuals( itpl_method( param)))

    parameters
    ----------
    pixel_list : list of pixel-obj
    param : parameter used for interpolation method
    param_name : name of parameter as used in the interpolation method, eg: "smooth"
    cv_args: arguments for .cv_itpl (excluding 'method_args')
    method_args : args passed from .cv_itpl to method (excluding `param`)
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(np.floor(os.cpu_count() * 0.75))) as executor:
        res_list = [executor.submit(
            _muliprocess_help, pix, cv_args=cv_args) for pix in pixel_list]
    # get result
    res_list = [res.result() for res in res_list]

    ##non - multiporcess
    # res_list = [pix.cv_itpl(**cv_args)["cv_res_"].to_numpy()
    #             for pix in pixel_list]
    # trim first and last observation (and convert to numpy)
    res_list = [res[1:(len(res) - 1)] for res in res_list]
    return res_list


def fun_to_optimize(statistic=quantile(0.5), res_list=None, param=None, param_name=None, cv_args=None, method_args=None, pixel_list=None):
    if (res_list is None) and (param is not None)\
            and (param_name is not None)\
            and (cv_args is not None)\
            and (method_args is not None)\
            and (pixel_list is not None):
        res_list = get_res_list(
            pixel_list, param, param_name, cv_args, method_args)
    return statistic([statistic(res) for res in res_list])
