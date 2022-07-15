# This contains pure interpolation function with
# itpl_fun(x, y, xx, weights, *args, **kwargs)

import numpy as np
import pandas as pd
import pykrige
import pickle
from scipy.signal import savgol_filter
from csaps import csaps  # for natural cubic smoothing splines
import scipy.interpolate as interpolate
import scipy.optimize  # for curve_fit
import my_utils.loess
import warnings
import my_utils.data_handle as data_handle

# get optimal parameter utility
try:  # (dont substitude this with data_handle)
    with open("data/computation_results/cv_itpl_res/optim_itpl_param", "rb") as f:
        optim_itpl_param = pickle.load(f)
except:
    optim_itpl_param = None
    print("optimal parameters have to still be found")

try:
    kriging_med_param = data_handle.load(
        "./data/computation_results/kriging_med_param.pkl")
except:
    kriging_med_param = None
    print("kriging-median parameters have to still be found")


def get_optim_itpl_param(fun_name, das_gdd, par_name, optim_statistic="quantile95"):
    """
    get optimal itpl param
    example: 
    'smoothing_spline', smooth, 'smooth', optim_statistic = 'quantile95'"""
    if len(das_gdd) == 5:  # convention "gdd75"
        optim_statistic = "quantile" + das_gdd[3:5]
        das_gdd = das_gdd[0:3]
    par_key = (
        "param_" + fun_name + "__" + das_gdd + "_" +
        par_name + "_" + optim_statistic
    )
    if par_key not in optim_itpl_param.keys():
        print("unkown optim_itpl_param key, aviable are: ")
        print(optim_itpl_param.keys())
    return float(optim_itpl_param[par_key][0])


def optimize_param_least_squares(fun, x, y, **opt_param):
    """
    optimizes function (f(x, parameters)-y)^2 and tries diffent tolerances
    """
    opt_param = {**opt_param, "maxfev": 100}
    opt_param = {**opt_param, "xtol": None}
    opt_param = {**opt_param, "ftol": None}
    for tol in [10**(-k) for k in [7, 6, 5, 4]]:
        opt_param["xtol"] = tol
        opt_param["ftol"] = tol
        try:
            with warnings.catch_warnings():
                # supress warnings of covariance to being estimated
                warnings.simplefilter("ignore")
                popt, pcov = scipy.optimize.curve_fit(
                    fun, x, y, **opt_param)
            return popt
        except:
            pass
    return None

# itpl-methods


def smoothing_spline(x, y, xx, weights, smooth=None, **kwargs):
    """
    calculates smoothing splines at 'step-sequence'
    smooth: Value in [0,1]
            0 corresponds to linear function (lambda=infty)
            1 corresponds to perfect fit (lambda=0)
    """
    if smooth is None:
        raise Exception("set smoothing parameter")
    elif ("gdd" in str(smooth)) or ("das" in str(smooth)):
        smooth = get_optim_itpl_param(
            "smoothing_spline", smooth, "smooth")
    return np.asarray(csaps(x, y, xx, weights=weights, smooth=smooth, **kwargs), dtype="float64")


def cubic_spline(x, y, xx, weights, **kwargs):
    """
    calculates cubic splines at 'step-sequence'
    uses smoothing_spline function with `smooth=0`
    """
    return np.asarray(csaps(x, y, xx, weights=weights, smooth=0, **kwargs), dtype="float64")


def b_spline(x, y, xx, weights, smooth=None):
    """
    Fits B-splines to determined knots
    smooth: Value in [0,infty)
            sum((w * (y - g))**2,axis=0) <= smooth
            where g(x) is the smoothed interpolation of (x,y).
            Larger s means more smoothing while smaller values
            of s indicate less smoothing.
    """
    if smooth is None:
        raise Exception("set smoothing parameter")
    elif ("gdd" in str(smooth)) or ("das" in str(smooth)):
        smooth = get_optim_itpl_param("b_spline", smooth, "smooth")
    t, c, k = interpolate.splrep(x, y, weights, s=smooth, k=3)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    return np.asarray(spline(xx), dtype="float64")


def ordinary_kriging(
    x, y, xx, weights, ok_args=None, return_parameters=False, **kwargs
):
    """
    ok_args : arguments for pykrige.OrdinaryKriging
        "variogram_parameters": [psill, range, nugget]
    """
    if ok_args is None:
        ok_args = {"variogram_model": "gaussian"}
    elif ok_args == "gdd":
        ok_args = {"variogram_model": "gaussian",
                   "variogram_parameters": list(kriging_med_param)}
    elif not isinstance(ok_args, dict):
        print(ok_args)
        raise Exception("ok args are not a dictionary")
    ok = pykrige.OrdinaryKriging(x, np.zeros(
        x.shape), y, exact_values=False, **ok_args, **kwargs)
    y_pred, _ = ok.execute("grid", xx, np.array([0.0]))
    y_pred = np.squeeze(y_pred)
    if return_parameters:
        return np.asarray(y_pred, dtype="float64"), ok
    else:
        return np.asarray(y_pred, dtype="float64")


def savitzky_golay(x, y, xx, weights, degree=3, **kwargs):
    """
    Fits Points according to the savicky golay filter with
    window :    Windowsize
    degree :    degree of local fitted polynomial
    """
    raise Exception(
        "not implemented, cannot deal with `gdd`, so use loess instead")
    # interpolate missing data
    time_series_interp = pd.Series(xx).interpolate(method="linear")

    # apply SavGol filter
    time_series_savgol = savgol_filter(
        time_series_interp, window_length=7, polyorder=2)
    print(
        "for some implementation see: https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data"
    )


def fourier(x, y, xx, weights, opt_param=None):
    """
    fits fourier of order two to the data,
    to increase chance of convergence of scipy.optimize.curve_fit set
    inital guess and bounds. Example:
    """

    def _fourier(t, period, a0, a1, a2, b1, b2):
        c = 2 * np.pi / period
        return (
            a1 * np.cos(c * 1 * t)
            + b1 * np.sin(c * 1 * t)
            + a0
            + a2 * np.cos(c * 2 * t)
            + b2 * np.sin(c * 2 * t)
        )

    if opt_param is None:
        raise Exception("set opt param to 'gdd', 'das' or  manually")
    elif opt_param == "gdd":
        opt_param = {"p0": [2000, 1, 1, 1, 1, 1],
                     "bounds": ([1200, -1, -5, -5, -5, -5], [4000, 2, 5, 5, 5, 5])}
    elif opt_param == "das":
        opt_param = {"p0": [350, 1, 1, 1, 1, 1],
                     "bounds": ([50, -1, -5, -5, -5, -5], [500, 2, 5, 5, 5, 5])}
    if weights is not None:
        # in the end the following is minimized:
        #   sum((residuals / sigma)^2)
        sigma = [np.sqrt(1 / w) for w in weights]
        opt_param = {**opt_param, "sigma": sigma}
    popt = optimize_param_least_squares(_fourier, x, y, **opt_param)
    if popt is None:
        return np.full(len(xx), np.nan)
    obj = np.asarray([_fourier(t, *popt) for t in xx])
    return np.asarray(obj, dtype="float64")


def double_logistic(x, y, xx, weights, opt_param=None, **kwargs):
    """
    fits double-logistic of order two to the data,
    to increase chance of convergence of scipy.optimize.curve_fit set
    inital guess and bounds. Example:
    opt_param={"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
        "bounds": ([0,0,0,10,0,-1], [1,1,300,300,1,0])})
    """

    def _double_logistic(t, ymin, ymax, start, duration, d0, d1):
        if not (np.any(t >= 0) and np.any(t < 1e+4) and ymin > -1 and ymin < 1 and ymax > -1 and ymax < 1 and
                start > -1 and start < 10000 and duration > -1 and duration < 10000 and
                d0 >= 0 and d0 <= 1 and d1 >= -1 and d1 <= 0):
            print("parameters:")
            print(t, ymin, ymax, start, duration, d0, d1)
            raise Exception("double logistic parameters are corrupt")
        # np.maximum(0,...) is to make overflow events less catastrophic
        return ymin + (ymax - ymin) * (
            1 / (1 + np.maximum(0, np.exp(-d0 * (t - start))))
            + 1 / (1 + np.maximum(0, np.exp(-d1 * (t - (start + duration)))))
            - 1
        )
    # # add artificially points outside the range to make optimization stable
    # n = len(x)
    # if n < 6:
    #     print(f"in doublelogistic only {n} observations - return nan's")
    #     return np.full(n, np.nan)
    # if n < 9:  # add artificially points
    #     print(f"in doublelogistic only {n} observations - extended borders")
    #     x = np.append(np.insert(x, 0, (x[0] - x[-1])), np.array(2 * x[-1]))
    #     y = np.append(np.insert(y, 0, (1 / 4)), np.array(2 * y[-1]))
    #     weights = np.append(
    #         np.insert(weights, 0, (weights[0])), np.array(weights[-1]))

    if opt_param is None:
        raise Exception("set opt param to 'gdd', 'das' or  manually")
    elif opt_param == "gdd":
        opt_param = {"p0": [0.2, 0.8, 800, 1400, 1 / 300, -1 / 300],
                     "bounds": ([0, 0.4, 100, 800, 0, -1 / 100], [0.7, 1, 1500, 3000, 1 / 100, 0])}
    elif opt_param == "das":
        opt_param = {"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
                     "bounds": ([0, 0, 0, 10, 0, -1], [1, 1, 300, 300, 1, 0])}
    if weights is not None:
        # in the end the following is minimized:
        #   sum((residuals / sigma)^2)
        sigma = [np.sqrt(1 / w) for w in weights]
        opt_param = {**opt_param, "sigma": sigma}
    popt = optimize_param_least_squares(_double_logistic, x, y,
                                        **opt_param, **kwargs)
    # print(popt)
    if popt is None:
        return np.full(len(xx), np.nan)
    obj = np.asarray([_double_logistic(t, *popt) for t in xx])
    return np.asarray(obj, dtype="float64")


def whittaker(x, y, xx, weights, *args, **kwargs):
    raise Exception(
        "smoothing splines are more general (can interpolate + can treat non-equidistant points)"
    )


## from moepy import lowess
# issue: strange behaviour in regions with little points
# def loess(x, y, xx, weights, alpha=0.25, robust=True, deg=2, **kwargs):
#     """
#     Calculation of the local regression coefficients for
#     a LOWESS model across the dataset provided. This method
#     will reassign the `frac`, `weighting_locs`, `loading_weights`,
#     and `design_matrix` attributes of the `Lowess` object.

#     Parameters:
#         x: values for the independent variable
#         y: values for the dependent variable
#         frac: LOWESS bandwidth for local regression as a fraction
#         reg_anchors: Locations at which to center the local regressions
#         num_fits: Number of locations at which to carry out a local regression
#         external_weights: Further weighting for the specific regression
#         robust_weights: Robustifying weights to remove the influence of outliers
#         robust_iters: Number of robustifying iterations to carry out
#     """
#     lowess_model = lowess.Lowess()
#     # default parameters for fit:
#     # frac=0.4, reg_anchors=None,
#     # num_fits = None, external_weights = None,
#     # robust_weights = None, robust_iters = 3, **reg_params
#     lowess_model.fit(x, y, frac=alpha, external_weights=weights, **kwargs)
#     return lowess_model.predict(xx)

# # import loess
# # issue : no weighting possible
# xout, yout, wout = loess.loess_1d(x, y, xnew=None, degree=1, frac=0.5,
#                                   npoints=None, rotate=False, sigy=None)


def loess(x, y, xx, weights, alpha=0.5, robust=False, deg=1):
    if ("gdd" in str(alpha)) or ("das" in str(alpha)):
        alpha = get_optim_itpl_param("loess", alpha, "alpha")
    # ensure alpha is big enough, i.e.:
    #     len(x) > alpha * len(x) > deg +1 (use 2 for extra security)
    alpha = np.min([1, np.max([alpha, (deg + 2) / len(x[weights > 0])])])
    yy = my_utils.loess.loess(
        x, y, alpha, xx=xx, poly_degree=deg, apriori_weights=weights, robustify=robust
    )[1].g.to_numpy()
    return np.asarray(yy, dtype="float64")


## import statsmodels.api as sm
# issue: does not support interpolation (only smoothing)
# def manual_loess(x, y, xx, weights, alpha=0.2, deg=2):
#     lowess = sm.nonparametric.lowess(y, x, frac=alpha)
#     # unpack the lowess smoothed points to their values
#     yy = np.empty(len(xx))
#     for i in range(len(y)):
#         yy[np.where(x[i]==xx)] = y[i]
#     sm.nonparametric.lowess(yy, xx, frac=alpha, return_sorted=False)

#     return lowess
