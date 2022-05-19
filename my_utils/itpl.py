# This contains pure interpolation function with
# itpl_fun(x, y, xx, weights, *args, **kwargs)

import numpy as np
import pandas as pd
import pykrige
from scipy.signal import savgol_filter
from csaps import csaps  # for natural cubic smoothing splines
import scipy.interpolate as interpolate
import scipy.optimize  # for curve_fit
import scipy.signal as ss  # for savitzky-Golayi
import my_utils.loess as loess


def smoothing_spline(x, y, xx, weights, smooth=None, **kwargs):
    """
    calculates smoothing splines at 'step-sequence'
    smooth: Value in [0,1]
            0 corresponds to linear function (lambda=infty)
            1 corresponds to perfect fit (lambda=0)
    """
    if smooth is None:
        raise Exception("set smoothing parameter")
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
    t, c, k = interpolate.splrep(x, y, weights, s=smooth, k=3)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    obj = spline(xx)
    return


def ordinary_kriging(x, y, xx, weights, ok_args=None, **kwargs):
    """
    ok_args : arguments for pykrige.OrdinaryKriging
        "variogram_parameters": [psill, range, nugget]
    """
    if ok_args is None:
        ok_args = {"variogram_model": "gaussian"}
    ok = pykrige.OrdinaryKriging(x, np.zeros(
        x.shape), y, exact_values=False, **ok_args)
    y_pred, y_std = ok.execute("grid", xx, np.array([0.0]))
    return np.squeeze(y_pred)


def savitzky_golay(x, y, xx, weights, degree=3, **kwargs):
    """
    Fits Points according to the savicky golay filter with
    window :    Windowsize
    degree :    degree of local fitted polynomial
    """
    # interpolate missing data
    time_series_interp = pd.Series(xx).interpolate(method="linear")

    # apply SavGol filter
    time_series_savgol = savgol_filter(
        time_series_interp, window_length=7, polyorder=2)
    print("for some implementation see: https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data")
    raise Exception(
        "not implemented, difficulty to extraploate (estimate value in between of two other values)")


def fourier(x, y, xx, weights=None, opt_param=None):
    """
    fits fourier of order two to the data,
    to increase chance of convergence of scipy.optimize.curve_fit set
    inital guess and bounds. Example:
    opt_param={"p0": [350, 1, 1, 1, 1, 1],
        "bounds": ([50, -1, -5, -5, -5, -5], [500, 2, 5, 5, 5, 5])})
    """
    def _fourier(t, period, a0, a1, a2, b1, b2):
        c = 2 * np.pi / period
        return a1 * np.cos(c * 1 * t) + b1 * np.sin(c * 1 * t) + \
            a0 + a2 * np.cos(c * 2 * t) + b2 * np.sin(c * 2 * t)
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


def double_logistic(x, y, xx, weights=None, opt_param=None):
    """
    fits double-logistic of order two to the data,
    to increase chance of convergence of scipy.optimize.curve_fit set
    inital guess and bounds. Example:
    opt_param={"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
        "bounds": ([0,0,0,10,0,-1], [1,1,300,300,1,0])})
    """
    def _double_logistic(t, ymin, ymax, start, duration, d0, d1):
        return ymin + (ymax - ymin) * (1 / (1 + np.exp(-d0 * (t - start))) + 1 / (1 + np.exp(-d1 * (t - (start + duration)))) - 1)
    if opt_param is None:
        opt_param = {}
    if weights is not None:
        # in the end the following is minimized:
        #   sum((residuals / sigma)^2)
        sigma = [np.sqrt(1 / w) for w in weights]
        opt_param = {**opt_param, "sigma": sigma}
    popt, pcov = scipy.optimize.curve_fit(
        _double_logistic, x, y, **opt_param)
    print(popt)
    obj = [_double_logistic(t, *popt) for t in xx]
    return np.asarray(obj)


def whittaker(x, y, xx, weights, *args, **kwargs):
    raise Exception("no weighting implemented yet")
    return None


def loess(x, y, xx, weights=None, alpha=0.25, robust=True, deg=2):
    raise Exception("no weighting implemented yet")
    return loess.loess(x, y, alpha=alpha, poly_degree=deg, robustify=True)