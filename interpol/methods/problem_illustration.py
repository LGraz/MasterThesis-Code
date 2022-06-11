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

x_axis = "das"

kriging_med_param = data_handle.load(
    "./data/computation_results/kriging_med_param.pkl")
method_strategy_label_kwargs = [
    # , ("get_cubic_spline", "cubic_spline", {})
    # , ("get_savitzky_golay", "savitzky_golay", {})
    (itpl.smoothing_spline, strategies.identity_no_xtpl,
     "smoothing_spline", {"smooth": x_axis}),
    (itpl.loess, strategies.identity_no_xtpl, "loess", {"alpha": x_axis}),
    (itpl.b_spline, strategies.identity_no_xtpl,
     "b_spline", {"smooth": x_axis}),
    (itpl.ordinary_kriging, strategies.identity_no_xtpl, "ordinary_kriging", {"ok_args": {
     "variogram_model": "gaussian", "variogram_parameters": list(kriging_med_param)}}),
    (itpl.fourier, strategies.identity_no_xtpl, "fourier", {"opt_param": {"p0": [350, 1, 1, 1, 1, 1], "bounds": (
        [50, -1, -5, -5, -5, -5], [500, 2, 5, 5, 5, 5])}}),
    (itpl.double_logistic, strategies.identity_no_xtpl,
     "double_logistic", {"opt_param": x_axis})
]
my_utils.plot.plot_3x3_pixels(method_strategy_label_kwargs, x_axis=x_axis)
# plt.show()

plt.savefig('../latex/figures/interpol/problem_illustration.pdf',
            bbox_inches='tight')

# %%


def bla(method, method_name, method_shortname, par_name, x_axis="gdd"):
    method_strategy_label_kwargs = []
    for times in [0, 1, 2, 3, 4]:
        method_strategy_label_kwargs.append(
            (method, strategies.robust_reweighting,
             f"{method_shortname} {times}-reweighted",
             {"update": False, par_name: x_axis, "times": times,
              "multiply_negative_res": 2}))
    my_utils.plot.plot_3x3_pixels(method_strategy_label_kwargs, x_axis=x_axis)
    plt.gcf().suptitle(f"{method_name}, iteratively reweighted")


# Smoothing splines
bla(itpl.smoothing_spline, "Smoothing splines", "SS", "smooth")

# Loess
bla(itpl.loess, "Loess,", "loess", "alpha")

# Double logistic
bla(itpl.double_logistic, "Double Logistic", "DL", "opt_param")

# B-splines
bla(itpl.b_spline, "B-splines", "B-Splines", "smooth")

# %%
x_axis = "gdd"
method_strategy_label_kwargs = []
for quantile in ["50", "75", "85", "90", "95"]:
    method_strategy_label_kwargs.append(
        (itpl.smoothing_spline, strategies.identity_no_xtpl,
         "quantile " + quantile, {"smooth": x_axis + quantile})
    )

my_utils.plot.plot_3x3_pixels(method_strategy_label_kwargs, x_axis=x_axis)
plt.gcf().suptitle(
    f"Smoothing Splines with parameter optimized w.r.t. quantile(" + quantile + ")")


# %%
