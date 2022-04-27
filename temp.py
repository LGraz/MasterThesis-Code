# %%
from cProfile import label
import numpy as np
import os
import sys
import importlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import my_utils.plot_settings

while "methods" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.cv as cv
import my_utils.data_handle as data_handle
import my_utils.pixel as pixel
importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel

# %%

np.random.seed(123)
pixels = data_handle.get_pixels(0.0008)
pix = pixels[0]

t = np.arange(0, 4, step=0.5)

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
pix.plot_ndvi("o")
pix.get_double_logistic(opt_param={"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
                                   "bounds": ([0, 0, 0, 10, 0, -1], [1, 1, 300, 300, 1, 0])})
pix.plot_step_interpolate("dl", label="double logistic")
pix.get_smoothing_spline(smooth=0.01)
pix.plot_step_interpolate("ss", label="smoothing splines")
plt.legend()
plt.title("PDF plot")
plt.savefig('../latex/figures/interpol/test.pdf')


# pdf = PdfPages('../latex/figures/interpol/test.pdf')
# pdf.savefig(fig)
# pdf.close()

#
#  same with .PGF
#
matplotlib.use("pgf")
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
pix.plot_ndvi("o")
pix.get_double_logistic(opt_param={"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
                                   "bounds": ([0, 0, 0, 10, 0, -1], [1, 1, 300, 300, 1, 0])})
pix.plot_step_interpolate("dl", label="double logistic")
pix.get_smoothing_spline(smooth=0.01)
pix.plot_step_interpolate("ss", label="smoothing splines")
plt.legend()
plt.title("PGF plot")
plt.savefig('../latex/figures/interpol/test_pfg.pgf')
