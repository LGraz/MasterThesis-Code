"""
    Illustrate various methods using the same 9 (3x3) pixels

    1. First problem illustration

    2. Robust Iterative Least Squares - comparison
    
    3. Comparing different statistics
       (used for parameter optimization)
    """
# %%
import os
import sys
import matplotlib.pyplot as plt

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.data_handle as data_handle
import my_utils.itpl as itpl
import my_utils.strategies as strategies
import my_utils.plot
import my_utils.plot_settings

##########################################
# 1. First problem illustration
##########################################
x_axis = "gdd"

method_strategy_label_kwargs = [
    (itpl.smoothing_spline, strategies.identity_no_xtpl,
     "smoothing_spline", {"smooth": x_axis}),
    (itpl.loess, strategies.identity_no_xtpl, "loess", {"alpha": x_axis}),
    (itpl.b_spline, strategies.identity_no_xtpl,
     "b_spline", {"smooth": x_axis}),
    (itpl.ordinary_kriging, strategies.identity_no_xtpl,
     "ordinary_kriging", {"ok_args": "gdd"}),
    (itpl.fourier, strategies.identity_no_xtpl,
     "fourier", {"opt_param": "gdd"}),
    (itpl.double_logistic, strategies.identity_no_xtpl,
     "double_logistic", {"opt_param": x_axis})
]
my_utils.plot.plot_3x3_pixels(
    method_strategy_label_kwargs, x_axis=x_axis, pixels=my_utils.plot.pixels_2x3)
# plt.show()

plt.savefig('../latex/figures/interpol/problem_illustration.pdf',
            bbox_inches='tight')

# %%

##########################################
# 2. Robust Iterative Least Squares - comparison
##########################################


def bla(method, method_name, method_shortname, par_name, x_axis="gdd", pixels=my_utils.plot.pixels_2x3):
    """using `method` we interpolate 9 pixels by robustifierng and 
    compare iterations"""
    method_strategy_label_kwargs = []
    for times in [0, 1, 2, 3, 4]:
        method_strategy_label_kwargs.append(
            (method, strategies.robust_reweighting,
             f"{method_shortname} {times}-reweighted",
             {"update": False, par_name: x_axis, "times": times,
              "multiply_negative_res": 2}))
    my_utils.plot.plot_3x3_pixels(
        method_strategy_label_kwargs, x_axis=x_axis, pixels=pixels)
    ## plt.gcf().suptitle(f"{method_name}, iteratively reweighted")
    # save figure
    plt.savefig(f'../latex/figures/interpol/2x3_{method_shortname}_robust.pdf',
                bbox_inches='tight')


# Loess
bla(itpl.loess, "Loess,", "loess", "alpha")

# Smoothing splines
bla(itpl.smoothing_spline, "Smoothing splines", "SS", "smooth")

# Double logistic
bla(itpl.double_logistic, "Double Logistic", "DL", "opt_param")

# B-splines
bla(itpl.b_spline, "B-splines", "B-Splines", "smooth")

# %%
##########################################
# 3. Comparing different statistics
#    (used for parameter optimization)
##########################################
x_axis = "gdd"
method_strategy_label_kwargs = []
for quantile in ["50", "75", "85", "90", "95"]:
    method_strategy_label_kwargs.append(
        (itpl.smoothing_spline, strategies.identity_no_xtpl,
         "quantile " + quantile, {"smooth": x_axis + quantile})
    )

my_utils.plot.plot_3x3_pixels(
    method_strategy_label_kwargs, x_axis=x_axis, pixels=my_utils.plot.pixels_2x3)
plt.savefig('../latex/figures/interpol/statistics_SS_param_optim.pdf',
            bbox_inches='tight')


# %%
