import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from csaps import csaps  # for natural cubic smoothing splines
import scipy.interpolate as interpolate


class Pixel:
    """
    Attributes
    ----------
    'coord_id' : coordinate of the Pixel
    'cov' : data extracted from Covariates.csv
    'cov_date_np' : numpy array with `cov.date` converted to unix date
    'cov_n' : number of rows of `cov`
    'met' : weather data from pixel (big range of date)
    'ndvi' : spectral index calculated from `cov`
    'step' : timestep between interpolated sequence
    'step_interpolate' : DataFrame with interpolated seqences, has two collumns with date (`date-format` and `unix-format`)
    'yie : data from `yield.csv`
    """

    def __init__(self, d_cov, d_met, d_yie, coord_id="random", step=24*3600):
        if coord_id == "random":
            coord_id = d_cov.coord_id.to_frame().sample(
                1, ignore_index=True).coord_id[0]
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
        """
        get NDVI := NIR(Band8)-Red(Band4)/NIR(Band8)+Red(Band4)
        """
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

    def _init_step_interpolate(self):
        """
        initialize object where interpolation-sequences are going to be stored
        also convertes dates into usable sequences
        """
        if hasattr(self, "step_interpolate"):
            raise Exception("step_interpolate has already been set")
        xs_np, xs_pd = self.get_unix_date_sequence()
        self.step_interpolate = pd.DataFrame(
            {"date": xs_pd, "date_unix": xs_np})

    def _prepare_interpolation(self, name, y=None, ind_keep=None):
        """
        preprocessing for interpolation

        Parameters
        ----------
        name:   the name of collumn in the `step_interpolate`
        y:      what we interpolate 'against', be default the NDVI is used

        Returns
        -------
        x:      unix-formatted dates of observations
        y:      values of observations
        xs_np:  unix-formatted equidistant sequence of dates (first to last date), with: `delta t` = `step`  
        """
        if not (hasattr(self, "step_interpolate") & hasattr(self, "cov_date_np")):
            self._init_step_interpolate()
        if ind_keep is None:
            ind_keep = [True]*self.cov_n
        if y is None:
            y = self.get_ndvi()
        if name in self.step_interpolate.columns:
            raise Exception("There already exists an collumn named: " + name)
        x = self.cov_date_np[ind_keep]
        y = y[ind_keep]
        xs_np = self.step_interpolate.date_unix
        return x, y, xs_np

    def get_smooting_spline(self, y=None, smooth=0.1, name="ss", ind_keep=None):
        """
        calculates smoothing splines at 'step-sequence'
        smooth: Value in [0,1]
                0 corresponds to linear function (lambda=infty)
                1 corresponds to perfect fit (lambda=0)
        """
        x, y, xs_np = self._prepare_interpolation(name, y, ind_keep)
        const = 1  # e-11/(28*24*3600)
        obj = csaps(x, y, xs_np, smooth=smooth * const)
        obj = pd.DataFrame(obj, columns=[name])
        self.step_interpolate = self.step_interpolate.join(obj)
        return obj

    def get_cubic_spline(self, y=None, name="cubic_spline", ind_keep=None):
        """
        calculates cubic splines at 'step-sequence'
        uses smoothing_spline function with `smooth=0`
        """
        return self.get_smooting_spline(y=y, smooth=0, name=name, ind_keep=ind_keep)

    def get_b_spline(self, y=None, name="BSpline", smooth=0.1, ind_keep=None):
        """
        Fits B-splines to determined knots
        smooth: Value in [0,infty)
                sum((w * (y - g))**2,axis=0) <= smooth 
                where g(x) is the smoothed interpolation of (x,y). 
                Larger s means more smoothing while smaller values 
                of s indicate less smoothing. 
        """
        x, y, xs_np = self._prepare_interpolation(name, y, ind_keep)
        t, c, k = interpolate.splrep(x, y, s=smooth, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        obj = spline(xs_np)
        obj = pd.DataFrame(obj, columns=[name])
        self.step_interpolate = self.step_interpolate.join(obj)
        return obj

    def plot_step_interpolate(self, which="ss"):
        if which not in self.step_interpolate.columns:
            raise Exception(
                "*which* is not a collumn in self.step_interpolate")
        # self.step_interpolate.plot(kind="line", x="date", y=which)
        x = self.step_interpolate.date
        y = self.step_interpolate[which]
        plt.plot(x, y)

    def get_unix_date_sequence(self):
        """
        Function which helps with different time-formats
           Converts pandas 'dateSeries' to unix time and provides
           unix-numpy and pandas series with ´step´-seconds
         Output:
           'pd_date in unix-numpy array',
           'unix-numpy series with `step`-increase',
           'pandas-series with `step`-increase'
         Default: increase of one day

        convert to unix
        """
        x = pd.to_datetime(self.cov.date).astype(int) / 10**9
        x = x.to_numpy()
        # get equaliy spaced dates
        xs_np = np.arange(x.min(), x.max(), self.step)  # each day
        # convert from unix to %Y-%m-%d
        xs_pd = pd.DataFrame(xs_np * 10**9)
        xs_pd = pd.to_datetime(xs_pd[0], format="%Y-%m-%d")
        self.cov_date_np = x
        return xs_np, xs_pd

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


###################### END Pixel ########################


# def random_pixel(d_cov, d_met, d_yie, n=1):
#     result = []
#     cid = d_cov.coord_id.to_frame().sample(n, ignore_index=True).coord_id
#     for i in range(n):
#         result.append(pixel(cid[i], d_cov, d_met, d_yie))
#     return result

# def unix_date_seqence(pd_date, step=24*3600):
# # Function which helps with different time-formats
# #    Converts pandas 'dateSeries' to unix time and provides
# #    unix-numpy and pandas series with ´step´-seconds
# #  Output:
# #    'pd_date in unix-numpy array',
# #    'unix-numpy series with `step`-increase',
# #    'pandas-series with `step`-increase'
# #  Default: increase of one day
#     # convert to unix
#     x = pd.to_datetime(pd_date).astype(int) / 10**9
#     x = x.to_numpy()
#     # get equaliy spaced dates
#     xs_np = np.arange(x.min(), x.max(), step)  # each day
#     # convert from unix to %Y-%m-%d
#     xs_pd = pd.DataFrame(xs_np * 10**9)
#     xs_pd = pd.to_datetime(xs_pd[0], format="%Y-%m-%d")
#     return x, xs_np, xs_pd
