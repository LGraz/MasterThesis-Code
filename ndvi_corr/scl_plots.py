"""
    Create 9 plots (one for each scl_class)

    plot estimated (out of bag, and using only scl in [4,5]) ndvi
    VS.
    observed ndvi (from space)
    """
# %%
import itertools
import matplotlib.pyplot as plt
import pandas as pd

ndvi_table = pd.read_pickle(
    "./data/computation_results/ndvi_tables/ndvi_table_all_years_01.pkl"
)

ax_inds = itertools.product([0, 1, 2], [0, 1, 2])
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
for class_nr, ax_ind in zip([2, 3, 4, 5, 6, 7, 8, 9, 10], ax_inds):
    plt.sca(ax[ax_ind[0], ax_ind[1]])

    scl_description = {
        0: "No Data (Missing data on projected tiles)",
        1: "Saturated or defective pixel",
        2: "Dark features / Shadows",
        3: "Cloud shadows",
        4: "Vegetation",
        5: "Bare soils / deserts",
        6: "Water",
        7: "Cloud low probability",
        8: "Cloud medium probability",
        9: "Cloud high probability",
        10: "Thin cirrus",
        11: "Snow or ice",
    }
    plt.axis("square")
    plt.title(f"SCL class {class_nr}: {scl_description[class_nr]}")
    ind = ndvi_table.scl_class == class_nr
    plt.scatter(
        ndvi_table.ndvi_observed[ind], ndvi_table.ndvi_itpl_ss_noex[ind], s=2, alpha=0.5
    )
    plt.xlim([0, 1])
    plt.xlabel("observed ndvi")
    plt.ylim([0, 1])
    plt.ylabel("estimated ndvi (with scl in [4,5])")
    plt.plot((0, 1), (0, 1), c="black", alpha=0.4)

print("figure generated, saving ...")
plt.savefig(
    "../latex/figures/ndvi_corr/scl_residuals_scatter.png", dpi=200, bbox_inches="tight"
)
