"""
    Construct plots of Satelite-image-time-series with witzwil-background
    for a choosen pixel (the same one used for the NDVI-correction-demo)
"""
# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
while "plot" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

import my_utils.plot_settings
import my_utils.data_handle as data_handle
import my_utils.plot


border_padding = 20
savefig = True
fig_dir = "../latex/figures/satelite/time_series_2021_P112"


# get pixel
pix = data_handle.get_pixels(
    0.001, cloudy=True, train_test="train", seed=4321)[13]
dates = np.unique(pix.cov.date.to_numpy())
if not savefig:
    print(dates)

# read date
cov_test = data_handle.read_df(
    "./data/yieldmapping_data/cloudy_data/yearly_train_test_sets/2021_cereals_covariates_test.csv")
cov_train = data_handle.read_df(
    "./data/yieldmapping_data/cloudy_data/yearly_train_test_sets/2021_cereals_covariates_train.csv")
cov = pd.concat([cov_test, cov_train], axis=0).reset_index(drop=True)
cov


class WitzwilFieldImage:
    """contains one fild of Witzwil for one date"""

    def __init__(self, cov: pd.DataFrame, FID):
        self.cov = cov.loc[cov.FID == FID]
        self.FID = FID

    def __str__(self):
        return self.FID + "-------" + str(self.cov.date.to_numpy()[0])

    def __repr__(self):
        return self.__str__()

    def plot(self, type="rgb", highlight=True, **kwargs):
        """plot field on sentinel2-resolution with response determined by `type`

        Args:
            type (str, optional): what should be colord against. Defaults to "rgb".
            highlight (bool, optional): should the selected pixel be highlighted 
                in red? Defaults to True.

        Returns:
            _type_: data-frame-row of the highlighted pixel
        """
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.axis("off")

        # set xlim & ylim
        xx = self.cov.x_coord.to_numpy()
        yy = self.cov.y_coord.to_numpy() - 10
        xxmin = np.min(xx) - border_padding
        xxmax = np.max(xx) + border_padding + 10
        yymin = np.min(yy) - border_padding
        yymax = np.max(yy) + border_padding + 10
        xmin_old, xmax_old = ax.get_xlim()
        ymin_old, ymax_old = ax.get_ylim()
        # ignore if default values (0 and 1)
        xmin_old = xxmin if xmin_old == 0 else xmin_old
        ymin_old = yymin if ymin_old == 0 else ymin_old
        xmax_old = xxmax if xmax_old == 1 else xmax_old
        ymax_old = yymax if ymax_old == 1 else ymax_old
        ax.set_xlim([min(xxmin, xmin_old), max(xxmax, xmax_old)])
        ax.set_ylim([min(yymin, ymin_old), max(yymax, ymax_old)])

        # add rectangles to `patches` and plot them
        patches = []
        highlighted = None
        for index, row in self.cov.iterrows():
            # (x,y) describe the top-left of the pixel
            x = row.x_coord
            y = row.y_coord - 10
            if type == "rgb":
                color = (row.B04 / 4000, row.B03 / 4000, row.B02 / 4000)
                color = tuple((max(min(i, 1), 0)
                              for i in color))  # ensure (0,1)
            elif type == "ndvi":
                ndvi = ((row.B08 - row.B04) / (row.B08 + row.B04))
                ndvi = max(0, ndvi)
                # color = plt.get_cmap("Greys")(ndvi)
                color = plt.get_cmap("Greens")(ndvi)
            else:
                color = "blue"
            # append rectangles to patches, but ensure that highlighted, is appended last
            if highlight and (row.coord_id == pix.coord_id):
                highlighted = matplotlib.patches.Rectangle(
                    (x, y), 10.0, 10.0, ec="red", fc=color, **kwargs)
                highlighted_row = row
            else:
                patches.append(matplotlib.patches.Rectangle(
                    (x, y), 10.0, 10.0, ec=color, fc=color, **kwargs))
        if highlighted is not None:
            patches.append(highlighted)
        # ax.add_collection(PatchCollection(patches, cmap=plt.get_cmap("plasma")))
        for patch in patches:
            ax.add_patch(patch)
        return highlighted_row


class WitzwilImage:
    """describes all witzwil-fields at a current date
    """

    def __init__(self, cov: pd.DataFrame, date):
        if date not in np.unique(cov.date.to_numpy()):
            raise Exception("date not in covariates")
        self.cov = cov.loc[cov.date == date]
        self.date = date
        FIDs = np.unique(self.cov.FID.to_numpy())
        self.fields = [WitzwilFieldImage(self.cov, FID) for FID in FIDs]

    def plot(self, **kwargs):
        for field in self.fields:
            field.plot(**kwargs)


def plot_satelite_image(date, plot_pix_ndvi=False, type="rgb"):
    # preperation
    if plot_pix_ndvi:  # plot ndvi-ts on the right hand
        # raise Exception("Dont do this, this is deprecrated, better plot \
        #     seperately and do mulicol in latex")
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=[
            6, 2.2], gridspec_kw={'width_ratios': [1.5, 1]})
        plt.sca(ax1)
        date_index = int(np.where(dates == date)[0][0])
        pix.plot_ndvi(ind=date_index + 1, colors="scl45_grey")
        plt.sca(ax0)
    else:
        plt.figure(figsize=[6, 6])
    ax = plt.gca()

    # plot satelite picture
    field = WitzwilFieldImage(WitzwilImage(cov, date).cov, pix.FID.iloc[0])
    row = field.plot(type=type)
    # add background-image
    img = plt.imread("../latex/figures/satelite/witzwil_2021_P112.png")
    img_extent = [351392.0, 352310.0, 5204125.0, 5205050.0]
    ax.imshow(img, extent=img_extent)
    return row


print("NOW PLOT AND SAVE IMAGES ---------------------------------")
for i in tqdm(range(len(dates))):
    # for i in tqdm([32]):
    try:
        date_index = i
        date = dates[date_index]

        # plot sateliteimage
        row = plot_satelite_image(date)
        if savefig:
            plt.savefig(
                fig_dir + "/" + f"{date_index:02d}_scl{row.scl_class}_" + date + ".png", bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        # plot ndvi timeseries
        plt.figure(figsize=[5, 2])
        pix.plot_ndvi(ind=date_index + 1, colors="scl45_grey")
        if savefig:
            plt.savefig(fig_dir + "_ndvi/" +
                        f"{date_index:02d}_scl{row.scl_class}_" + date + ".pdf", bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        # plot both
        row = plot_satelite_image(date, plot_pix_ndvi=True)
        plt.gca().set_title(f"scl class: {row.scl_class}")
        if savefig:
            plt.savefig(fig_dir + "+ndvi/" +
                        f"{date_index:02d}_scl{row.scl_class}_" + date + ".pdf", bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    except Exception as e:
        print(e)
