# interpolation strategys (like iterative procedures / reweighting ...)
# meant to be used by: pixel_obj.itpl(strategy, ...)
import scipy.stats
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from wcwidth import wcswidth


def identity(itpl_method, x, y, xx, w, *args, **kwargs):
    return itpl_method(x, y, xx, w, *args, **kwargs)


def identity_no_xtpl(itpl_method, x, y, xx, w, *args, **kwargs):
    """
    same as identity, but outside of the training bounds (x,y) take the same
    value at the corresponding
    --> this should fix cases where interpol-method 'explodes' outside of the
    data-range
    """
    yy = itpl_method(x, y, xx, w, *args, **kwargs)
    x_min = np.min(x)
    ind = np.where(xx <= x_min)[0]
    if ind.size > 0:
        yy_x_min = yy[np.max(ind)]
        yy[ind] = yy_x_min
    x_max = np.max(x)
    ind = np.where(xx >= x_max)[0]
    if ind.size > 0:
        yy_x_max = yy[np.min(ind)]
        yy[ind] = yy_x_max
    return yy


def weighted_median(arr, w):
    ind = np.where(w > 0)  # lets not devide later with 0
    arr = arr[ind]
    w = w[ind]
    order = np.argsort(arr)
    arr = arr[order]
    w = w[order]
    w_cumsum = np.cumsum(w)
    w_cumsum = w_cumsum / w_cumsum[-1]  # sum(w) != 1
    i_lower = np.max(np.where(w_cumsum <= 0.5))
    i_upper = np.min(np.where(w_cumsum >= 0.5))
    return (arr[i_lower] * w[i_lower] + arr[i_upper] * w[i_upper]) / (2 * (w[i_upper] + w[i_lower]))


def robust_reweighting(itpl_method, x, y, xx, w, *args, times=3,
                       debug=False, multiply_negative_res=1, **kwargs):
    """
    times : how often refit & reweigthed
    multiply_negative_res : factor, with which the negative residuals are multiplied
    """
    # debug = kwargs.pop("debug", None)
    # times = kwargs.pop("times", None)
    if times == 0:
        return itpl_method(x, y, xx, w, *args, **kwargs)
    else:
        # predict y (only on x-grid, not xx-grid)
        y_pred = itpl_method(x, y, x, w, *args, **kwargs)
        res = y - y_pred
        res[res < 0] = res[res < 0] * multiply_negative_res
        # get weighed mad
        sigma = weighted_median(np.abs(res - weighted_median(res, w)), w)
        # NDVI noise shall be no smaller than some threshold
        sigma = np.max([sigma, 0.1 / 6])
        if debug:
            print(f"sigma = {sigma}")
        # apply biweight loss
        res = res / (6 * sigma)
        w = [np.max([1 - res[i]**2, 0]) * np.max([1 -
                                                  res[i]**2, 0]) for i in range(len(x))]
        w = np.array(w)
        if debug:
            print(f"weights = {w}")
        ind = np.where(w > 0)
        # recursion
        result = robust_reweighting(itpl_method, x[ind], y[ind], xx, w[ind],
                                    *args, times=times - 1,
                                    multiply_negative_res=multiply_negative_res,
                                    debug=debug, **kwargs)
        result = np.asarray(result, dtype="float64")
        # return previous iteration if only nan's produced
        if np.sum(np.isnan(result)) == len(result):
            return y_pred
        else:
            return result


def cv(itpl_method, x, y, xx, w, *args,
       cv_strategy=identity, return_residuals=False,
       **kwargs):
    k = len(x)
    kf = KFold(k, shuffle=True)
    y_oob_pred = np.empty(len(y))  # out of bag predictions
    y_oob_pred.fill(np.nan)
    for train, test in kf.split(x):
        x_train = np.array(x[train])
        y_train = np.array(y[train])
        x_test = np.array(x[test])
        w_train = np.array(w[train])
        y_test_pred = cv_strategy(
            itpl_method, x_train, y_train, x_test,
            w_train, *args, **kwargs)
        y_oob_pred[test] = y_test_pred

    # put results in xx--format
    result = np.empty(len(xx))
    result.fill(np.nan)
    temp = set(x)  # for faster computing
    xx_ind_in_x = np.array(
        [i for i in range(len(xx)) if xx[i] in temp])
    xx_ind_in_x = np.intersect1d(xx_ind_in_x, np.unique(
        xx, return_index=True)[1])  # omit equal `gdd` values
    if return_residuals:
        result[xx_ind_in_x] = y - y_oob_pred
    else:
        result[xx_ind_in_x] = y_oob_pred
    return result
