# %%
import os
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
from my_utils.cv import get_pix_cv_resiudals
import my_utils.data_handle as data_handle
import my_utils.pixel as pixel
import my_utils.strategies as strategies
import my_utils.itpl as itpl
from my_utils.pixel_multiprocess import pixel_multiprocess
importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(strategies)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel


pixels = data_handle.get_pixels(0.1, seed=4321, cloudy=True)
res = pixel_multiprocess(
    pixels, get_pix_cv_resiudals, itpl.smoothing_spline,
    cv_strategy=strategies.identity_no_xtpl, par_name="smooth",
    par_value="gdd")
print(np.min(res), np.max(res))

# %%
from scipy import stats

ag, bg = stats.laplace.fit(res)
xx = np.linspace(-0.4, 0.4, num=500)
plt.plot(xx, stats.laplace.pdf(xx, ag, bg))
plt.hist(res, bins=100, range=(-0.4, 0.4), density=True)
plt.title("Histogram of Interpolation-Residuals (smoothing splines)")
plt.savefig('../latex/figures/interpol/res_cv.pdf',
            bbox_inches='tight')

# %%
