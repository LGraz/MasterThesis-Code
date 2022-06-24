"""
plots a nice and a bad fourier and double-logistic plot
"""
# %%
import os
import sys
from xml.etree.ElementTree import PI
import matplotlib.pyplot as plt
while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.data_handle as data_handle
import my_utils.plot_settings
from my_utils.pixel import Pixel
import my_utils.itpl as itpl
import my_utils.strategies as strategies

pixels = data_handle.get_pixels(0.005, seed=123)


def plot_fourier_and_doublelogistic(pix: Pixel):
    # pix.x_axis = "das"
    pix.itpl("fourier", itpl.fourier, strategies.identity, opt_param="gdd")
    pix.plot_ndvi(ylim=[0, 1])
    pix.plot_itpl_df("fourier", label="2nd order fourier")
    pix.itpl("dl", itpl.double_logistic, strategies.identity, opt_param="gdd")
    pix.plot_itpl_df("dl", label="double logistic")
    plt.legend(loc="lower center")


ratio = 0.55

plt.subplot(1, 2, 1)  # index 2
plt.title("Expected Behaviour")
plot_fourier_and_doublelogistic(pixels[62])
my_utils.plot_settings.set_plot_ratio(ratio)

plt.subplot(1, 2, 2)  # row 1, col 2 index 1
plt.title("Degenerated Example")
plot_fourier_and_doublelogistic(pixels[11])
my_utils.plot_settings.set_plot_ratio(ratio)

plt.savefig('../latex/figures/interpol/fourier_dl_comparison.pdf',
            bbox_inches='tight')

# %%
i = 0
for pix in pixels:
    plot_fourier_and_doublelogistic(pix)
    plt.title(str(i))
    plt.show()
    i = i + 1
# %%
