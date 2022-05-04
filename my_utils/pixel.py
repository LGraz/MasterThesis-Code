import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pykrige
from sklearn.model_selection import KFold
from scipy.signal import savgol_filter

from csaps import csaps  # for natural cubic smoothing splines
import scipy.interpolate as interpolate
import scipy.optimize  # for curve_fit
import scipy.signal as ss  # for savitzky-Golayi


class Pixel:
    """
    Attributes
    ----------
    'coord_id' : coordinate of the Pixel
    'cov' : data extracted from Covariates.csv
    'cov_date_np' : numpy array with `cov.date` converted to unix date
    'cov_n' : number of rows of `cov` for this pixel
    'met' : weather data from pixel (big range of date)
    'ndvi' : spectral index calculated from `cov`
    'step' : timestep between interpolated sequence
    'step_interpolate' : DataFrame with interpolated seqences, has two collumns with date (`date-format` and `unix-format`)
    'yie : data from `yield.csv`
    """

    def __init__(self, d_cov, d_yie, d_met=None, coord_id="random", step=1, use_date=False):
        """
        Init size max: 0.4 MB (if all years are considerd)
        """
        if coord_id == "random":
            coord_id = d_cov.coord_id.to_frame().sample(
                1, ignore_index=True).coord_id[0]
        self.coord_id = coord_id
        self.cov = d_cov[d_cov.coord_id == coord_id]
        self.cov_n = len(self.cov)
        if self.cov_n < 4:
            raise Exception(
                f"Pixel {coord_id} has not enough observations (less then 4)"
            )
        self.yie = d_yie[d_yie.coord_id == coord_id]
        self.FID = self.cov.FID  # can take instead: set(...)
        if not (d_met is None):
            self.met = d_met[d_met.FID.isin(set(self.FID))]
        self.use_date = use_date  # use day after sawing, otherwise
        # only one year per pixel !
        x = pd.to_datetime(self.cov.date).astype(int) / (10**9 * 24 * 3600)
        if (x.max() - x.min()) > 365:
            raise Exception(
                "Pixel carry more information for more than a year")
        self.step = step  # timestep used for interpolation in days

# printing method:
    def __str__(self):
        return "FID:  " + str(set(self.FID)) + "--------------------------" + "\n" + "yield: " + str(self.yie) + "\n" + "coord_id: " + self.coord_id + "\n"

    def __repr__(self):
        return self.__str__()

# utils
    def get_ndvi(self):
        """
        get NDVI := NIR(Band8)-Red(Band4)/NIR(Band8)+Red(Band4)
        """
        if not hasattr(self, 'ndvi'):
            self.ndvi = (self.cov.B08 - self.cov.B04) / \
                (self.cov.B08 + self.cov.B04)
        return self.ndvi

# init interpolation
    def set_step(self, step):
        if hasattr(self, "step_interpolate"):
            raise Exception("*step* has been already used to get other things")
        self.step = step

    def get_unix_date_sequence(self):
        """
        update: now unix*24*3600*1000 provided

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
        x = pd.to_datetime(self.cov.date).astype(int) / (10**9 * 24 * 3600)
        x = x.to_numpy()
        # get equaliy spaced dates
        xs_np = np.arange(x.min(), x.max() + 1, self.step)  # each day
        # convert from unix to %Y-%m-%d
        xs_pd = pd.DataFrame(xs_np * (10**9 * 24 * 3600))
        xs_pd = pd.to_datetime(xs_pd[0], format="%Y-%m-%d")
        self.cov_date_np = x
        return xs_np, xs_pd

    def _init_step_interpolate(self):
        """
        initialize object where interpolation-sequences are going to be stored
        also convertes dates into usable sequences
        """
        if hasattr(self, "step_interpolate"):
            raise Exception("step_interpolate has already been set")
        xs_np, xs_pd = self.get_unix_date_sequence()
        # for some reason the seqence starts one day after the first observation
        # and ends one day before the last one
        self.step_interpolate = pd.DataFrame(
            {"date": xs_pd, "date_unix": xs_np, "das": self.cov.das.iloc[0] + range(len(xs_pd))})

    def _prepare_interpolation(self, name, y=None, ind_keep=None):
        """
        preprocessing for interpolation

        Parameters
        ----------
        name:   the name of collumn in the `step_interpolate`
        y:      what we interpolate 'against', be default the NDVI is used
        ind_keep: list of boolean of length cov.n

        Returns
        -------
        x:      unix-formatted dates of observations or days after sawing
        y:      values of observations
        time:  unix-formatted equidistant sequence of dates (first to last date), with: `delta t` = `step`
        """
        if not (hasattr(self, "step_interpolate")):
            self._init_step_interpolate()
        if ind_keep is None:
            ind_keep = [True] * self.cov_n
        if len(ind_keep) != self.cov_n:
            raise Exception("ind_keep of wrong length")
        if y is None:
            y = self.get_ndvi()
        if name in self.step_interpolate.columns:
            # raise Exception("There already exists an collumn named: " + name)
            print("There already exists an collumn named: " + name)
        if self.use_date:
            x = self.cov_date_np[ind_keep]
            time = self.step_interpolate.date_unix
        else:
            x = self.cov.das[ind_keep]
            time = self.step_interpolate.das
        y = y[ind_keep]
        return x, y, time

# interpolation
    def get_smoothing_spline(self, y=None, name="ss", ind_keep=None, save_data=True, smooth=None):
        """
        calculates smoothing splines at 'step-sequence'
        smooth: Value in [0,1]
                0 corresponds to linear function (lambda=infty)
                1 corresponds to perfect fit (lambda=0)
        """
        if smooth is None:
            raise Exception("set smoothing parameter")
        x, y, time = self._prepare_interpolation(name, y, ind_keep)
        obj = csaps(x, y, time, smooth=smooth)
        obj = pd.DataFrame(obj, columns=[name])
        if save_data:
            if name in self.step_interpolate.columns:
                self.step_interpolate[name] = obj.to_numpy()
            else:
                self.step_interpolate = self.step_interpolate.join(obj)
        return obj

    def get_cubic_spline(self, y=None, name="cubic_spline", ind_keep=None, save_data=True):
        """
        calculates cubic splines at 'step-sequence'
        uses smoothing_spline function with `smooth=0`
        """
        return self.get_smoothing_spline(y=y, smooth=0, name=name, ind_keep=ind_keep, save_data=save_data)

    def get_b_spline(self, y=None, name="BSpline", ind_keep=None, save_data=True, smooth=0.1):
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
        if save_data:
            if name in self.step_interpolate.columns:
                self.step_interpolate[name] = obj.to_numpy()
            else:
                self.step_interpolate = self.step_interpolate.join(obj)
        return obj

    def get_ordinary_kriging(self, y=None, name="OK", ind_keep=None, save_data=True, ok_args=None):
        """
        ok_args : arguments for pykrige.OrdinaryKriging
            "variogram_parameters": [psill, range, nugget]
        """
        x, y, time = self._prepare_interpolation(name, y, ind_keep)
        if ok_args is None:
            ok_args = {"variogram_model": "gaussian"}
        ok = pykrige.OrdinaryKriging(x, np.zeros(
            x.shape), y, exact_values=False, **ok_args)
        y_pred, y_std = ok.execute("grid", time, np.array([0.0]))
        y_pred = np.squeeze(y_pred)
        # y_std = np.squeeze(y_std)
        obj = pd.DataFrame(y_pred, columns=[name])
        if save_data:
            if name in self.step_interpolate.columns:
                self.step_interpolate[name] = obj.to_numpy()
            else:
                self.step_interpolate = self.step_interpolate.join(obj)
        return obj, ok

    def get_savitzky_golay(self, y=None, name="savitzky_golay", ind_keep=None, window=5, degree=3):
        """
        Fits Points according to the savicky golay filter with
        window :    Windowsize
        degree :    degree of local fitted polynomial
        """
        x, y, xs_np = self._prepare_interpolation(name, y, ind_keep)
        print("for some implementation see: https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data")
        # create a random time series
        time_series = np.random.random(50)
        time_series[time_series < 0.1] = np.nan
        time_series = pd.Series(time_series)

        # interpolate missing data
        time_series_interp = time_series.interpolate(method="linear")

        # apply SavGol filter
        time_series_savgol = savgol_filter(
            time_series_interp, window_length=7, polyorder=2)
        raise Exception(
            "not implemented, difficulty to extraploate (estimate value in between of two other values)")

    def get_fourier(self, y=None, name="fourier", ind_keep=None, save_data=True, weights=None, opt_param=None):
        """
        fits fourier of order two to the data,
        to increase chance of convergence of scipy.optimize.curve_fit set
        inital guess and bounds. Example:
        opt_param={"p0": [350, 1, 1, 1, 1, 1],
            "bounds": ([50, -1, -5, -5, -5, -5], [500, 2, 5, 5, 5, 5])})
        """
        x, y, time = self._prepare_interpolation(name, y, ind_keep)

        def fourier(t, period, a0, a1, a2, b1, b2):
            c = 2 * np.pi / period
            return a1 * np.cos(c * 1 * t) + b1 * np.sin(c * 1 * t) + \
                a0 + a2 * np.cos(c * 2 * t) + b2 * np.sin(c * 2 * t)
        if opt_param is None:
            opt_param = {}
        if weights is not None:
            # in the end the following is minimized:
            #   sum((residuals / sigma)^2)
            sigma = [np.sqrt(1 / w) for w in weights]
            opt_param = {**opt_param, "sigma": sigma}
        popt, pcov = scipy.optimize.curve_fit(fourier, x, y, **opt_param)
        print(popt)
        obj = [fourier(t, *popt) for t in time]
        obj = pd.DataFrame(obj, columns=[name])
        if save_data:
            if name in self.step_interpolate.columns:
                self.step_interpolate[name] = obj.to_numpy()
            else:
                self.step_interpolate = self.step_interpolate.join(obj)
        return obj, popt

    def get_double_logistic(self, y=None, name="dl", ind_keep=None, save_data=True, weights=None, opt_param=None):
        """
        fits double-logistic of order two to the data,
        to increase chance of convergence of scipy.optimize.curve_fit set
        inital guess and bounds. Example:
        opt_param={"p0": [0.2, 0.8, 50, 100, 0.01, -0.01],
            "bounds": ([0,0,0,10,0,-1], [1,1,300,300,1,0])})
        """
        x, y, time = self._prepare_interpolation(name, y, ind_keep)

        def double_logistic(t, ymin, ymax, start, duration, d0, d1):
            return ymin + (ymax - ymin) * (1 / (1 + np.exp(-d0 * (t - start))) + 1 / (1 + np.exp(-d1 * (t - (start + duration)))) - 1)
        if opt_param is None:
            opt_param = {}
        if weights is not None:
            # in the end the following is minimized:
            #   sum((residuals / sigma)^2)
            sigma = [np.sqrt(1 / w) for w in weights]
            opt_param = {**opt_param, "sigma": sigma}
        popt, pcov = scipy.optimize.curve_fit(
            double_logistic, x, y, **opt_param)
        print(popt)
        obj = [double_logistic(t, *popt) for t in time]
        obj = pd.DataFrame(obj, columns=[name])
        if save_data:
            if name in self.step_interpolate.columns:
                self.step_interpolate[name] = obj.to_numpy()
            else:
                self.step_interpolate = self.step_interpolate.join(obj)
        return obj, popt

# cross validation
    def _init_cv_interpolate(self):
        if not hasattr(self, "step_interpolate"):
            self._init_step_interpolate()
        self.cv_interpolate = pd.DataFrame(
            {"date": self.step_interpolate.date, "date_unix": self.step_interpolate.date_unix, "das": self.step_interpolate.das})

    def cv_interpolation(self, methodname="", y=None, k=5, method='get_smoothing_spline', method_args={}):
        """
        Description
        -----------
        Perform k fold crossvaldation and returns residuals

        Parameters
        ----------
        method : sth like "get_smoothing_spline"
        methodname : short name for method (used for naming collumn) eg: "ss" for smoothing spline
        y : target variable
        k : number of folds, set k=np.inf for LOOCV
        method_args : list with arguments for method

        Returns
        -------
        DataFrame with residuals

        from now on apply further functions which calculate statistics
        like RMSE, Med, MAD, QN, ....
        """
        one_result_column = True
        if k is None:
            k = self.cov_n
        if y is None:
            y = self.get_ndvi()
        k = min(self.cov_n, k)
        kf = KFold(k, shuffle=True)
        x = self.cov.das
        residuals = pd.DataFrame(
            {"das": x, ("cv_" + methodname + "_truth"): y})
        if not hasattr(self, "cv_interpolate"):
            self._init_cv_interpolate()
        iter = -1
        if one_result_column:
            cv_res_name = "cv_res_" + methodname
            residuals[cv_res_name] = np.nan
        for train, test in kf.split(x):
            iter += 1
            ind_keep_bool = [i in train for i in range(int(self.cov_n))]
            name = "cv_" + methodname + "_" + str(iter)
            cv_name = "cv_" + name + "_" + str(iter)
            args = {**method_args, 'ind_keep': ind_keep_bool,
                    'name': cv_name, "save_data": False}
            obj = getattr(self, method)(**args)
            # locate ind
            ind_test_bool = [i in test for i in range(self.cov_n)]
            # ind_test_bool_for_step_interpolate
            ind_test = [i in x[ind_test_bool].to_numpy()
                        for i in self.step_interpolate.das]
            obj = obj[ind_test].set_index(y.iloc[test].index)
            # calculate residuals
            res = y.iloc[test] - getattr(obj, cv_name)
            if one_result_column:
                residuals[cv_res_name] = residuals[cv_res_name].add(
                    res, fill_value=0)
            else:
                res = pd.DataFrame(
                    res, columns=["res_" + methodname + "_" + str(iter)])
                residuals = residuals.join(res)
        return residuals

# plot
    def plot_step_interpolate(self, which="ss", *args, **kwargs):
        if which not in self.step_interpolate.columns:
            raise Exception(
                "*which* is not a collumn in self.step_interpolate")
        # self.step_interpolate.plot(kind="line", x="date", y=which)
        if self.use_date:
            x = self.step_interpolate.date
        else:
            x = self.step_interpolate.das
        y = self.step_interpolate[which]
        plt.plot(x, y, *args, **kwargs)

    def plot_ndvi(self, *args, ylim=None, **kwargs):
        if not hasattr(self, 'ndvi'):
            self.get_ndvi()
        if self.use_date:
            x = pd.to_datetime(self.cov.date)
        else:
            x = self.cov.das
        plt.plot(x, self.ndvi, *args, **kwargs)
        plt.ylabel("NDVI")
        if not self.use_date:
            plt.xlabel("DAS")
        if ylim is None:
            plt.ylim([0, 1])
        else:
            plt.ylim(ylim)
        # for showing only some dates this might be helpful:
        # https://www.geeksforgeeks.org/matplotlib-figure-figure-autofmt_xdate-in-python/
        plt.gcf().autofmt_xdate()
# Filter observations

    def filter_ndvi_min(self, date, i):
        if i in [0, len(self.cov) - 1]:
            return True
        else:
            return (self.ndvi.iloc[i - 1] < self.ndvi.iloc[i]) | (self.ndvi.iloc[i] > self.ndvi.iloc[i + 1])

    def filter_method(self, method, date, i):
        if method == "ndvi_min":
            return self.filter_ndvi_min(date, i)
        else:
            print("filter method unkown")
            return False
        # match method:
        #     case "ndvi_min":
        #         return self.filter_ndvi_min(date, i)
        #     case _:
        #         print("filter method unkown")
        #         return False

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
