#%%
import os
import sys
import matplotlib.pyplot as plt

while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

import my_utils.plot_settings
from my_utils.data_handle import get_pixels

pix = get_pixels(0.001, cloudy=True, train_test="train", seed=4321)[13]

plt.rcParams.update({"font.size": 16})
for x_axis in ["das", "gdd"]:
    pix.x_axis = x_axis
    plt.figure()
    plt.rcParams.update({"font.size": 16})
    pix.plot_ndvi(colors="scl45")
    plt.savefig(f"../latex/figures/interpol/ndvi_ts_{x_axis}.pdf", bbox_inches="tight")

plt.figure()
plt.rcParams.update({"font.size": 16})
my_utils.plot_settings.legend_scl45_grey()

for x_axis in ["das", "gdd"]:
    pix.x_axis = x_axis
    pix.plot_ndvi(colors="scl45_grey")
    plt.savefig(
        f"../latex/figures/interpol/ndvi_ts_{x_axis}_grey.pdf", bbox_inches="tight"
    )
    plt.figure()
    plt.rcParams.update({"font.size": 16})

#%%
for cols in ["scl", "scl45_grey"]:
    if cols == "scl45_grey":
        plt.figure()
        my_utils.plot_settings.legend_scl45_grey(fontsize=14)
    elif cols == "scl":
        plt.figure()
        my_utils.plot_settings.legend_scl(fontsize=16, bbox_to_anchor=(1, 1))

    plt.rcParams.update({"font.size": 16})
    pix.plot_ndvi(colors=cols)
    date_dict = {
        "2021-02-23": "(a)",
        "2021-05-09": "(b)",
        "2021-05-24": "(c)",
        "2021-06-03": "(d)",
        "2021-06-28": "(e)",
        "2021-07-23": "(f)",
    }

    for i in range(len(pix.cov.date)):
        gdd = pix.cov.gdd.to_numpy()[i]
        ndvi = pix.get_ndvi()[i]
        date = pix.cov.date.to_numpy()[i]
        if date in date_dict.keys():
            plt.text(gdd - 75, ndvi + 0.03, date_dict[date])

    # if cols == "scl":
    # plt.xlim([0,4300])

    plt.savefig(f"../latex/figures/interpol/ndvi_ts_{cols}.pdf", bbox_inches="tight")
# %%
