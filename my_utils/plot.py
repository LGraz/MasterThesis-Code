"""
create a 3x3 plot with 
"""
import matplotlib.pyplot as plt
import random
import itertools
import my_utils.data_handle as data_handle

pixels = data_handle.get_pixels(0.01, seed=4321)
random.seed(4321)
pixels_3x3 = random.sample(pixels, 30)
pixels_3x3 = [pixels_3x3[i] for i in [2, 1, 7, 8, 9, 10, 12, 13, 14]]


def plot_3x3_pixels(method_strategy_label_kwargs, x_axis="gdd", pixels=pixels_3x3):
    """
    creates a 3x3 plot of interpolations 
    """
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 10))
    ax_inds = itertools.product([0, 1, 2], [0, 1, 2])
    # plt.suptitle("Different NDVI-Series Examples")
    for pix, ax_ind in zip(pixels, ax_inds):
        pix.x_axis = x_axis
        plt.sca(ax[ax_ind[0], ax_ind[1]])
        pix.plot_ndvi()
        for itpl_method, itpl_stratgety, label, kwargs in method_strategy_label_kwargs:
            pix.itpl(label, itpl_method, itpl_stratgety, **kwargs)
            pix.plot_itpl_df(label, label=label)
    plt.sca(ax[2, 2])
    plt.gca().legend(loc="lower left", fontsize="large")
