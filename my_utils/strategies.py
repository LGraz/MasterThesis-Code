# interpolation strategys (like iterative procedures / reweighting ...)
# meant to be used by: pixel_obj.itpl(strategy, ...)
import scipy.stats
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd


def identity(itpl_method, x, y, xx, weights, *args, **kwargs):
    return itpl_method(x, y, xx, weights, *args, **kwargs)


def identity_no_extrapol(itpl_method, x, y, xx, weights, *args, **kwargs):
    """
    same as identity, but outside of the training bounds (x,y) take the same
    value at the corresponding
    --> this should fix cases where interpol-method 'explodes' outside of the
    data-range
    """
    yy = itpl_method(x, y, xx, weights, *args, **kwargs)
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


def robust_reweighting(itpl_method, x, y, xx, weights, *args, times=3,
                       debug=False, punish_negative=1, **kwargs):
    """
    times : how often refit & reweigthed
    punish_negative : factor, with which the negative residuals are multiplied
    """
    # debug = kwargs.pop("debug", None)
    # times = kwargs.pop("times", None)
    if times == 0:
        return itpl_method(x, y, xx, weights, *args, **kwargs)
    else:
        # predict y (only on x-grid, not xx-grid)
        y_pred = itpl_method(x, y, x, weights, *args, **kwargs)
        residuals = y - y_pred
        residuals[residuals < 0] = residuals[residuals < 0] * \
            punish_negative
        sigma = scipy.stats.median_abs_deviation(residuals)
        # NDVI noise shall be no smaller than some threshold
        sigma = np.max([sigma, 0.01])
        if debug:
            print(f"sigma = {sigma}")
        # apply biweight loss
        residuals = residuals / (6 * sigma)
        weights = [np.max([1 - residuals[i]**2, 0]) * np.max([1 -
                                                              residuals[i]**2, 0]) for i in range(len(x))]
        weights = np.array(weights)
        if debug:
            print(f"weights = {weights}")
        ind = np.where(weights > 0)
        # recursion
        return robust_reweighting(itpl_method, x[ind], y[ind], xx, weights[ind],
                                  *args, times=times - 1,
                                  punish_negative=punish_negative,
                                  debug=debug, **kwargs)


def cv(itpl_method, x, y, xx, weights, *args,
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
        weights_train = np.array(weights[train])
        y_test_pred = cv_strategy(
            itpl_method, x_train, y_train, x_test,
            weights_train, *args, **kwargs)
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