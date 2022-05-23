import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

import my_utils.itpl as itpl


class Pixel:
    """
    Attributes
    ----------
    'coord_id' : coordinate of the Pixel
    'cov' : data extracted from Covariates.csv
    'cov_n' : number of rows of `cov` for this pixel
    'met' : weather data from pixel (big range of dates)
    'ndvi' : spectral index calculated from `cov`
    'step' : timestep between interpolated sequence
    'itpl_df' : DataFrame with interpolated seqences, has two collumns with `das` and `gdd`
    'yie : data from `yield.csv`
    """

    def __init__(self, d_cov, d_yie, d_met=None, coord_id="random", x_axis="gdd", year=None):
        """
        Init size max: 0.4 MB (if all years are considerd)

        Parameters:
        -----------
        x_axis : what should the date be, possible values are "gdd" or "das"  (GrowingDegreeDays or DaysAfterSawing)
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
        self.x_axis = x_axis  # use day after sawing, otherwise

        # only one year per pixel !
        x = pd.to_datetime(self.cov.date).astype(int) / (10**9 * 24 * 3600)
        self.year = year
        if (x.max() - x.min()) > 365:
            raise Exception(
                "Pixel carry more information for more than a year")

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
            self.ndvi = ((self.cov.B08 - self.cov.B04) /
                         (self.cov.B08 + self.cov.B04)).to_numpy()
        return self.ndvi

# init interpolation
    def _init_itpl_df(self):
        """
        initialize self.itpl_df where `das`- and `gdd`- 
        interpolation-sequences are going to be stored
        """
        if hasattr(self, "itpl_df"):
            raise Exception("itpl_df has already been set")
        das = self.cov.das.to_numpy()
        a, b = (das[0], das[len(das) - 1])
        das_itpl_seq = np.linspace(a, b, num=b - a + 1).astype(int)
        gdd = self.cov.gdd.to_numpy()
        gdd_itpl_seq = np.round(
            np.interp(das_itpl_seq, das, gdd)).astype(int)
        gdd, gdd_itpl_seq
        self.itpl_df = pd.DataFrame(
            {"das": das_itpl_seq, "gdd": gdd_itpl_seq})

    def _prepare_itpl(self, name, y=None):
        """
        preprocessing for interpolation

        Parameters
        ----------
        name:   the name of collumn in the `itpl_df`
        y:      what we interpolate 'against', be default the NDVI is used

        Returns
        -------
        x:      `das` or `gdd` for each observation
        y:      values of observations (NDVI by default)
        xx:     x but (linearly) interpolated for each day
        """
        if not (hasattr(self, "itpl_df")):
            self._init_itpl_df()
        if name in self.itpl_df.columns:
            print("There already exists an collumn named: " + name)
        if y is None:
            y = self.get_ndvi()
        x = self.cov[self.x_axis].to_numpy()
        xx = self.itpl_df[self.x_axis].to_numpy()
        if len(x) != len(y):
            raise Exception("lengths of x and y do not match")
        return x, y, xx

# filter/weighting methods
    def filter_scl(self, weights, classes=[4, 5]):
        scl = self.cov.scl_class.to_numpy()
        weights[~np.isin(scl, classes)] = 0
        return weights

# interpolation strategys (like iterative procedures / reweighting ...)
    def strategy_identity(itpl_method, x, y, xx, weights, *args, **kwargs):
        return itpl_method(x, y, xx, weights, *args, **kwargs)

# interpolation
    def itpl(self, name, itpl_fun, itpl_strategy=strategy_identity,
             filter_method_kwargs=[("filter_scl", {"classes": [4, 5]})], **kwargs):
        """
        parameters
        ----------
        name : string to save results in `self.itpl_df` 
        itpl_fun : a interpolation-function arguments (x, y, xx, weights)
        itpl_strategy : a function which applies `itpl_fun`
        filter_method_kwargs : a list of tupel("filter_name", {**filter_kwargs}).
            specifies filtermethod and its argumets
            to apply several filtermethods mind the order
        **kwargs : kwargs which are passed down to iterpol_method
        """
        # prepare
        x, y, xx = self._prepare_itpl(name)

        # apply filter / weighting methods
        weights = np.asarray(([1] * len(x)))
        for filter_method, filter_kwargs in filter_method_kwargs:
            weights = getattr(self, filter_method)(weights, **filter_kwargs)

        # perform calcultions
        ind = np.where(weights > 0)
        result = itpl_strategy(
            itpl_fun, x[ind], y[ind], xx, weights[ind], **kwargs)

        # save result
        result_df = pd.DataFrame(result, columns=[name])
        if name in self.itpl_df.columns:
            self.itpl_df[name] = result_df.to_numpy()
        else:
            self.itpl_df = self.itpl_df.join(result_df)
        return result

# cross validation
    def _init_cv_itpl(self):
        if not hasattr(self, "itpl_df"):
            self._init_itpl_df()
        self.cv_itpl = pd.DataFrame(
            {"das": self.itpl_df.das, "gdd": self.itpl.gdd})

    def cv_itpl(self, methodname="", y=None, k=5, method='get_smoothing_spline', method_args={}):
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
        if not hasattr(self, "cv_itpl"):
            self._init_cv_itpl()
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
            # ind_test_bool_for_itpl_df
            ind_test = [i in x[ind_test_bool].to_numpy()
                        for i in self.itpl_df.das]
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
    def plot_itpl_df(self, which="ss", *args, **kwargs):
        if which not in self.itpl_df.columns:
            raise Exception(
                "*which* is not a collumn in self.itpl_df")
        if self.x_axis == "gdd":
            x = self.itpl_df.gdd
        elif self.x_axis == "das":
            x = self.itpl_df.das
        else:
            raise Exception("unknown x_axis")
        y = self.itpl_df[which]
        plt.plot(x, y, *args, **kwargs)

    def plot_ndvi(self, *args, ylim=None, scl_color=False, **kwargs):
        if not hasattr(self, 'ndvi'):
            self.get_ndvi()
        if self.x_axis == "gdd":
            x = self.cov.gdd
        elif self.x_axis == "das":
            x = self.cov.das
        else:
            raise Exception("unknown x_axis")
        if scl_color:
            cmap = {
                0: "#000000", 1: "#ff0000", 2: "#404040", 3: "#bf8144", 4: "#00ff3c", 5: "#ffed50",
                6: "#0d00fa", 7: "#808080", 8: "#bfbfbf", 9: "#eeeeee", 10: "#0bb8f0", 11: "#ffbfbf"}
            colors = list(map(float, self.cov.scl_class.tolist()))
            colors = [cmap[i] for i in colors]
            kwargs = {**kwargs, "c": colors}
        plt.scatter(x.tolist(), self.ndvi.tolist(), *args, **kwargs)
        plt.ylabel("NDVI")
        if self.x_axis == "gdd":
            plt.xlabel("GDD")
        elif self.x_axis == "das":
            plt.xlabel("DAS")
        else:
            raise Exception("unknown x_axis")
        if ylim is None:
            plt.ylim([0, 1])
        else:
            plt.ylim(ylim)

# for backwards-compatibility:
    def get_smoothing_spline(self, name="ss", **kwargs):
        return self.itpl(name, itpl.smoothing_spline, **kwargs)

    def get_cubic_spline(self, name="cubic_spline", **kwargs):
        return self.itpl(name, itpl.cubic_spline, **kwargs)

    def get_b_spline(self, name="BSpline", **kwargs):
        return self.itpl(name, itpl.b_spline, **kwargs)

    def get_ordinary_kriging(self, name="OK", **kwargs):
        return self.itpl(name, itpl.ordinary_kriging, **kwargs)

    def get_savitzky_golay(self, name="savitzky_golay", **kwargs):
        return self.itpl(name, itpl.savitzky_golay, **kwargs)

    def get_fourier(self, name="fourier", **kwargs):
        return self.itpl(name, itpl.fourier, **kwargs)

    def get_double_logistic(self, name="dl", **kwargs):
        return self.itpl(name, itpl.double_logistic, **kwargs)

    def get_whittaker(self, name="wt", **kwargs):
        return self.itpl(name, itpl.whittaker, **kwargs)

    def get_loess(self, name="loess", **kwargs):
        return self.itpl(name, itpl.loess, **kwargs)

###################### END Pixel ########################
