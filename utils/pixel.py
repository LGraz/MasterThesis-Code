import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from csaps import csaps  # for natural cubic smoothing splines


class pixel:
    def __init__(self, coord_id, d_cov, d_met, d_yie, step=24*3600):
        self.coord_id = coord_id
        self.cov = d_cov[d_cov.coord_id == coord_id]
        self.cov_n = len(self.cov)
        self.step = step  # timestep used for interpolation
        self.yie = d_yie[d_yie.coord_id == coord_id]
        self.FID = self.cov.FID  # can take instead: set(...)
        self.met = d_met[d_met.FID.isin(set(self.FID))]

    # printing method:
    def __str__(self):
        return "FID:  " + str(set(self.FID)) + "--------------------------" + "\n" + "yield: " + str(self.yie) + "\n" + "coord_id: " + self.coord_id + "\n"

    def __repr__(self):
        return self.__str__()

    def get_ndvi(self):
        # NDVI := NIR(Band8)-Red(Band4)/NIR(Band8)+Red(Band4)
        if not hasattr(self, 'ndvi'):
            self.ndvi = (self.cov.B08 - self.cov.B04) / \
                (self.cov.B08 + self.cov.B04)
        return self.ndvi

    def plot_ndvi(self):
        if not hasattr(self, 'ndvi'):
            self.get_ndvi()

        plt.plot(pd.to_datetime(self.cov.date), self.ndvi, "o")
        plt.ylabel("NDVI")
        plt.ylim([0, 1])
        # for showing only some dates this might be helpful:
        # https://www.geeksforgeeks.org/matplotlib-figure-figure-autofmt_xdate-in-python/
        plt.gcf().autofmt_xdate()
        # plt.show()

    def get_smooting_spline(self, y=None, smooth=0.1, name="ss", ind_keep=None):
        if not (hasattr(self, "step_interpolate") & hasattr(self, "cov_date_np")):
            self.init_step_interpolate()
        if ind_keep is None:
            ind_keep = [True]*self.cov_n
        if y is None:
            y = self.ndvi
        if name in self.step_interpolate.columns:
            raise Exception("There already exists an collumn named: " + name)
        x = self.cov_date_np[ind_keep]
        y = y[ind_keep]
        xs_np = self.step_interpolate.date_unix
        const = 1  # e-11/(28*24*3600)
        obj = csaps(x, y, xs_np, smooth=smooth * const)
        obj = pd.DataFrame(obj, columns=[name])
        self.step_interpolate = self.step_interpolate.join(obj)
        return ind_keep

    def plot_step_interpolate(self, which="ss"):
        if which not in self.step_interpolate.columns:
            raise Exception(
                "*which* is not a collumn in self.step_interpolate")
        # self.step_interpolate.plot(kind="line", x="date", y=which)
        x = self.step_interpolate.date
        y = self.step_interpolate[which]
        plt.plot(x, y)

    def get_unix_date_sequence(self):
        # Function which helps with different time-formats
        #    Converts pandas 'dateSeries' to unix time and provides
        #    unix-numpy and pandas series with ´step´-seconds
        #  Output:
        #    'pd_date in unix-numpy array',
        #    'unix-numpy series with `step`-increase',
        #    'pandas-series with `step`-increase'
        #  Default: increase of one day
        #
        # convert to unix
        x = pd.to_datetime(self.cov.date).astype(int) / 10**9
        x = x.to_numpy()
        # get equaliy spaced dates
        xs_np = np.arange(x.min(), x.max(), self.step)  # each day
        # convert from unix to %Y-%m-%d
        xs_pd = pd.DataFrame(xs_np * 10**9)
        xs_pd = pd.to_datetime(xs_pd[0], format="%Y-%m-%d")
        self.step_seq_np = xs_np
        self.step_seq_pd = xs_pd
        self.cov_date_np = x
        return xs_np, xs_pd

    def init_step_interpolate(self):
        if hasattr(self, "step_interpolate"):
            raise Exception("step_interpolate has already been set")
        xs_np, xs_pd = self.get_unix_date_sequence()
        self.step_interpolate = pd.DataFrame(
            {"date": xs_pd, "date_unix": xs_np})

    def set_step(self, step):
        if hasattr(self, "step_interpolate"):
            raise Exception("*step* has been already used to get other things")
        self.step = step

# Filter observations
    def filter_ndvi_min(self, date, i):
        if i in [0, len(self.cov)-1]:
            return True
        else:
            return (self.ndvi.iloc[i-1] < self.ndvi.iloc[i]) | (self.ndvi.iloc[i] > self.ndvi.iloc[i+1])

    def filter_method(self, method, date, i):
        match method:
            case "ndvi_min":
                return self.filter_ndvi_min(date, i)
            case _:
                print("filter method unkown")
                return False

    def filter(self, method):
        keep_ind = []
        for i, date in enumerate(self.cov.date):
            keep_ind.append(self.filter_method(method, date, i))
        self.keep_ind = keep_ind
        return keep_ind


###################### END PIXEL ########################
