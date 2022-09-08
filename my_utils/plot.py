"""
create a 3x3 plot with 
"""
import matplotlib
import matplotlib.pyplot as plt
import random
import itertools
import numpy as np

import my_utils.data_handle as data_handle
import my_utils.itpl as itpl
import my_utils.strategies as strategies
import my_utils.pixel as pixel
import my_utils.plot_settings

pixels = data_handle.get_pixels(0.01, seed=4321)
random.seed(4321)
pixels_3x3 = random.sample(pixels, 30)
pixels_3x3 = [pixels_3x3[i] for i in [2, 1, 7, 8, 9, 10, 12, 13, 14]]
pixels_2x3 = [pix for i, pix in enumerate(pixels_3x3) if i in [2, 3, 5, 6, 7, 8]]


# def set_size(w, h, ax=None):
#     """
#     Sets size of axis in figure
#      w, h: width, height in inches
#     https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
#     """
#     if not ax:
#         ax = plt.gca()
#     l = ax.figure.subplotpars.left
#     r = ax.figure.subplotpars.right
#     t = ax.figure.subplotpars.top
#     b = ax.figure.subplotpars.bottom
#     figw = float(w) / (r - l)
#     figh = float(h) / (t - b)
#     ax.figure.set_size_inches(figw, figh)


def plot_3x3_pixels(method_strategy_label_kwargs, x_axis="gdd", pixels=pixels_3x3):
    """
    creates a 3x3 plot of interpolations
    """
    plt.rcParams.update({"font.size": 9})
    if len(pixels) == 6:
        ncol = 3
        nrow = 2
    elif len(pixels) == 4:
        nrow = 2
        ncol = 3
    else:  # 3x3
        ncol = 3
        nrow = 3
    _, ax = plt.subplots(
        nrow, ncol, sharex=True, sharey=True, figsize=(ncol * 3, nrow * 2)
    )
    ax_inds = itertools.product(range(nrow), range(ncol))
    abc = "a"
    # plt.suptitle("Different NDVI-Series Examples")
    for pix, ax_ind in zip(pixels, ax_inds):
        pix.x_axis = x_axis
        plt.sca(ax[ax_ind[0], ax_ind[1]])
        pix.plot_ndvi(s=4)
        for itpl_method, itpl_stratgety, label, kwargs in method_strategy_label_kwargs:
            pix.itpl(label, itpl_method, itpl_stratgety, **kwargs)
            pix.plot_itpl_df(label, label=label, linewidth=0.9)
        plt.xlim([0,2600])
        timescale = pix.cov[x_axis].to_numpy()
        plt.text(40, 0.9, abc+")", fontsize="x-large")
        abc = chr(ord(abc) + 1) # increment abc-counter by 1 (from a -> b -> c ...)
    plt.sca(ax[nrow - 1, ncol - 1])
    plt.gca().legend(loc="lower left")


def plot_ndvi_corr_step(
    pix: pixel.Pixel,
    name,
    corr_method_name,
    corr_response,
    x_axis="gdd",
    transparancy=0.3,
    ind_leave_out=35,
    ax_list=None,
    refit_before_rob=False,
):
    """plot stepwise fitting and correction procedure for the ndvi
    this are several plots

    Args:
        pix (pixel.Pixel): _description_
        name (str): used in filename as identifier for plots
        model_ndvi (ml-model): ndvi_corrected = model_ndvi.predict(data)
        model_res : same as model_ndvi but resiuals
        covariates : which covariates shall be considered
        x_axis (str, optional): "gdd" or "das"
        transparancy (float, optional): ransparancy factor in plots
        ind_leave_out (int, optional): index of observation which shall be used
            as a demonstration for oob-ndvi-estimation
        ax_list  optional): list of axis where the plots shall go (not mandatory)
        refit_before_rob (bool, optional): also draw plot of last one (refittet)
    """
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 16,
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    if not pix.is_strictly_increasing_gdd():
        raise Exception("gdd not strictly increasing")

    pix.x_axis = x_axis
    y, res = pix.get_ndvi_corr(corr_method_name, corr_response)
    if True:
        w = np.full(len(y), 1)  # equal weights for now
        w = 1 / (np.abs(res))
    else:
        w = np.abs(res)
        w = -np.log(res / np.max(res))
    w = w * len(w) / np.sum(w)  # w shall be normed
    pix.itpl("ss", itpl.smoothing_spline, strategies.identity_no_xtpl, smooth=x_axis)
    rob_args = (itpl.smoothing_spline, strategies.robust_reweighting)
    rob_kwargs = {"y": y, "smooth": x_axis, "fit_strategy": strategies.identity_no_xtpl}

    i = 0
    for case in [
        "ndvi",
        "itpl",
        "itpl_rew",
        "ndvi_scl",
        "show_res",
        "corr",
        "uncert",
        "corr_itpl_rew",
    ]:
        i += 1
        plt.figure()  # plt.figure(figsize=[8,8])
        if ax_list is None:
            plt.sca = plt.gca()
        else:
            raise Exception("not implemented yet (axis_list")

        # NDVI45
        pix.plot_ndvi(
            colors="scl45", alpha=0 if case not in ["ndvi", "itpl", "itpl_rew"] else 1
        )
        if case == "ndvi":
            plt.savefig(
                "../latex/figures/step_plot/" + name + str(i) + "_" + case + ".pdf",
                bbox_inches="tight",
            )
            continue

        # First itpl
        pix.plot_itpl_df(
            "ss",
            alpha=transparancy * 1.6 if case not in ["itpl"] else 1,
            label="simple fit",
        )
        if case == "itpl":
            plt.legend()
            plt.savefig(
                "../latex/figures/step_plot/" + name + str(i) + "_" + case + ".pdf",
                bbox_inches="tight",
            )
            continue

        # # itpl - reweigthing
        # w_temp = np.full(len(pix.ndvi), 1)
        # pix.itpl("ss_rob45", *rob_args, w=w_temp, times=1, **rob_kwargs)
        # pix.plot_itpl_df(
        #     "ss_rob45",
        #     linewidth=2,
        #     alpha=transparancy * 2 if case not in ["itpl_rew", "ndvi_scl"] else 1,
        #     label="robust fit",
        # )
        if case == "itpl_rew":
            plt.legend()
            plt.savefig(
                "../latex/figures/step_plot/" + name + str(i) + "_" + case + ".pdf",
                bbox_inches="tight",
            )
            continue

        # NDVI ALL
        alpha_temp = transparancy * 1.2 if case not in ["ndvi_scl", "show_res"] else 1
        alpha_temp = 0 if "corr_itpl" in case else alpha_temp
        pix.plot_ndvi(colors="scl", alpha=alpha_temp)
        if case == "ndvi_scl":
            plt.legend()
            plt.savefig(
                "../latex/figures/step_plot/" + name + str(i) + "_" + case + ".pdf",
                bbox_inches="tight",
            )
            continue

        # residual illustration (out of bag)
        if case == "show_res":
            n = len(pix.ndvi)
            w_temp = np.full(n, 1)
            w_temp[ind_leave_out] = 0
            pix.itpl("ss_temp", *rob_args, **rob_kwargs, times=0, w=w_temp)
            pix.plot_itpl_df(
                "ss_temp", c="purple", alpha=0.4, linewidth=2, label="out-of-bag curve"
            )
            out_of_box_ndvi = pix.itpl_df["ss_temp"][
                pix.itpl_df["is_observation"]
            ].reset_index(drop=True)[ind_leave_out]
            x_temp = pix.cov[x_axis].reset_index(drop=True)[ind_leave_out]
            plt.scatter(x_temp, out_of_box_ndvi, c="red", s=100)
            plt.plot(
                [x_temp, x_temp], [pix.ndvi[ind_leave_out], out_of_box_ndvi], c="red"
            )
            plt.legend()
            plt.savefig(
                "../latex/figures/step_plot/" + name + str(i) + "_" + case + ".pdf",
                bbox_inches="tight",
            )
            continue

        # NDVI corretciontransparancy
        pix.plot_ndvi(colors="scl", corr=True)
        if case == "corr":
            plt.legend()
            plt.savefig(
                "../latex/figures/step_plot/" + name + str(i) + "_" + case + ".pdf",
                bbox_inches="tight",
            )
            continue

        # Uncertainty (errorbars)
        plt.errorbar(
            pix.cov[x_axis],
            pix.ndvi_corr,
            yerr=pix.ndvi_uncert * 2,
            fmt="none",
            c="black",
            alpha=transparancy if case not in ["uncert"] else 0.5,
        )
        if case == "uncert":
            plt.legend()
            plt.savefig(
                "../latex/figures/step_plot/" + name + str(i) + "_" + case + ".pdf",
                bbox_inches="tight",
            )
            continue

        # # robust refit
        # pix.itpl(
        #     "ss_rob",
        #     *rob_args,
        #     **rob_kwargs,
        #     w=w.copy(),
        #     times=1,
        #     filter_method_kwargs=[]
        # )
        # pix.plot_itpl_df("ss_rob", linewidth=2.5, label="corrected robust")

        if refit_before_rob:
            # refit (not robustified)
            pix.itpl(
                "ss_refit",
                *rob_args,
                **rob_kwargs,
                w=w.copy(),
                times=0,
                filter_method_kwargs=[]
            )
            pix.plot_itpl_df("ss_refit", linewidth=1, label="corrected")
        if case == "corr_itpl_rew":
            plt.legend()
            plt.savefig(
                "../latex/figures/step_plot/" + name + str(i) + "_" + case + ".pdf",
                bbox_inches="tight",
            )
            continue
    plt.legend()
