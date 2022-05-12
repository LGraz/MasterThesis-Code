# %%
import os
import sys
import numpy as np
import importlib
import matplotlib.pyplot as plt
while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

import my_utils.cv as cv
import my_utils.data_handle as data_handle
import my_utils.pixel as pixel
import my_utils.plot_settings

importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel

pixels = data_handle.get_pixels(0.001, cloudy=True, train_test="train")
# %%

pix = pixels[0]
# pix.plot_ndvi()
pix.cov.date
pix.cov.columns
x, y, t = pix._prepare_interpolation("hi")
x
# %%
# creates vector where every 5th observation (or less if clouds filtered) is not a nan
yy = np.empty(len(t))
yy[:] = np.nan
x = np.array(x)
temp = [i - x[0] for i in x]
yy[temp] = y
yy


# %%
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def savitzky_golay_filtering(timeseries, wnds=[7, 5], orders=[2, 4], debug=True):
    """ source:
    https://gis.stackexchange.com/questions/173721/reconstructing-modis-time-series-applying-savitzky-golay-filter-with-python-nump
    """
    interp_ts = pd.Series(timeseries).interpolate(method='linear', limit=9999)
    smooth_ts = interp_ts
    wnd, order = wnds[0], orders[0]
    F = 1e8
    it = 0
    while True:
        smoother_ts = savgol_filter(
            smooth_ts, window_length=wnd, polyorder=order)
        diff = (smoother_ts - interp_ts).clip(0)
        sign = diff > 0
        W = 1 - diff / np.max(diff)
        wnd, order = wnds[1], orders[1]
        fitting_score = np.sum(diff * W)
        print(it, ' : ', fitting_score)
        if fitting_score > F:
            break
        else:
            F = fitting_score
            it += 1
        smooth_ts = smoother_ts * sign + interp_ts * (1 - sign)
    if debug:
        return smooth_ts, interp_ts
    return smooth_ts


a, b = savitzky_golay_filtering(yy, debug=True)
pix.plot_ndvi()
plt.plot(t, pd.Series(yy).interpolate(method="linear"))
plt.plot(t, savitzky_golay_filtering(yy, debug=False))
np.max(a - b)
# %%
for pix in pixels:
    scl = pix.cov.scl_class
    ind = [s in [4, 5] for s in scl]
    x, y, t = pix._prepare_interpolation("hi", ind_keep=ind)
    yy = np.empty(len(t))
    yy[:] = np.nan
    x = np.array(x)
    temp = [i - x[0] for i in x]
    yy[temp] = y
    yy

    # pix.plot_ndvi()
    x2, y2, t2 = pix._prepare_interpolation("hi")
    for i, x_ in enumerate(x2):
        plt.text(x_, y2.to_numpy()[i], np.array(
            scl)[i], ha="center", va="center")
    plt.plot(t, pd.Series(yy).interpolate(method="linear"))
    plt.plot(t, savitzky_golay_filtering(yy, debug=False))
    plt.ylim([0, 1])
    np.max(a - b)
    plt.show()
