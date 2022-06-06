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
import scipy.signal as ss  # for savitzky-Golayi
import my_utils.loess
from scipy.interpolate import interp1d

# get optimal parameter utility
try:
    with open("data/computation_results/cv_itpl_res/optim_itpl_param", "rb") as f:
        optim_itpl_param = pickle.load(f)
except:
    optim_itpl_param = None
    print("optimal parameters have to still be found")


def get_optim_itpl_param(fun_name, das_gdd, par_name, quantile="75"):
    par_key = (
        "param_" + fun_name + "__" + das_gdd + par_name + "_" + "quantile" + quantile
    )
    return optim_itpl_param[par_key]


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
    elif smooth in ["gdd", "das"]:
        smooth = get_optim_itpl_param(
            "smoothing_spline", smooth, "smooth", "75")
    return csaps(x, y, xx, weights=weights, smooth=smooth, **kwargs)


def cubic_spline(x, y, xx, weights, **kwargs):
    """
    calculates cubic splines at 'step-sequence'
    uses smoothing_spline function with `smooth=0`
    """
    return csaps(x, y, xx, weights=weights, smooth=0, **kwargs)


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
    elif smooth in ["gdd", "das"]:
        smooth = get_optim_itpl_param("b_spline", smooth, "smooth", "75")
    t, c, k = interpolate.splrep(x, y, weights, s=smooth, k=3)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    return spline(xx)


def ordinary_kriging(
    x, y, xx, weights, ok_args=None, return_parameters=False, **kwargs
):
    """
    ok_args : arguments for pykrige.OrdinaryKriging
        "variogram_parameters": [psill, range, nugget]
    """
    if ok_args is None:
        ok_args = {"variogram_model": "gaussian"}
    ok = pykrige.OrdinaryKriging(x, np.zeros(
        x.shape), y, exact_values=False, **ok_args)
    y_pred, y_std = ok.execute("grid", xx, np.array([0.0]))
    y_pred = np.squeeze(y_pred)
    if return_parameters:
        return y_pred, ok
    else:
        return y_pred


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
    opt_param={"p0": [350, 1, 1, 1, 1, 1],
        "bounds": ([50, -1, -5, -5, -5, -5], [500, 2, 5, 5, 5, 5])})
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
        opt_param = {}
    if weights is not None:
        # in the end the following is minimized:
        #   sum((residuals / sigma)^2)
        sigma = [np.sqrt(1 / w) for w in weights]
        opt_param = {**opt_param, "sigma": sigma}
    popt, pcov = scipy.optimize.curve_fit(_fourier, x, y, **opt_param)
    obj = [_fourier(t, *popt) for t in xx]
    return np.asarray(obj)


def double_logistic(x, y, xx, weights, opt_param=None):
    """
    fits double-logistic of order two to the data,
    to increase chance of convergence of scipy.optimize.curve_fit set
    inital guess and bounds. Example:
    opt_param={"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
        "bounds": ([0,0,0,10,0,-1], [1,1,300,300,1,0])})
    """

    def _double_logistic(t, ymin, ymax, start, duration, d0, d1):
        return ymin + (ymax - ymin) * (
            1 / (1 + np.exp(-d0 * (t - start)))
            + 1 / (1 + np.exp(-d1 * (t - (start + duration))))
            - 1
        )

    if opt_param is None:
        opt_param = {}
    if weights is not None:
        # in the end the following is minimized:
        #   sum((residuals / sigma)^2)
        sigma = [np.sqrt(1 / w) for w in weights]
        opt_param = {**opt_param, "sigma": sigma}
    popt, pcov = scipy.optimize.curve_fit(_double_logistic, x, y, **opt_param)
    print(popt)
    obj = [_double_logistic(t, *popt) for t in xx]
    return np.asarray(obj)


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
    if alpha in ["gdd", "das"]:
        alpha = get_optim_itpl_param("loess", alpha, "alpha", "75")
    # ensure alpha is big enough, i.e.:
    #     len(x) > alpha * len(x) > deg +1 (use 2 for extra security)
    alpha = np.min([1, np.max([alpha, (deg + 2) / len(x[weights > 0])])])
    return my_utils.loess.loess(
        x, y, alpha, xx=xx, poly_degree=deg, apriori_weights=weights, robustify=robust
    )[1].g.to_numpy()


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
