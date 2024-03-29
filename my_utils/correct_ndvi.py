"""
    Description
        This file provides a bridge to use NDVI-correction 
        prediction models implemented in `R` in python. 

    Example:
        correct_ndvi(DataFrame_with_covariates, "rf", "ss_noex")
        "rf" : is the shortname shown in the dictionary below
        "ss_noex" : is the response (suffix) shown in the list below
            -> use "ss_noex_res" for residuals 
"""

# %%
import os
import numpy as np
import pandas as pd
import math
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

# convert pandas data frame to R-data frame
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

methods_without_response = {  # (predict_method_suffix, package, name)
    "rf": ("randomForest", "randomForest", "rf"),
    "lm_scl": ("lm", "stats", "ndvi_scl"),
    "lm_all": ("lm", "stats", "all"),
    # "lm_step": ("lm", "stats", "step"),
    "mars": ("earth", "earth", "mars"),
    "gam": ("gam", "mgcv", "gam"),
    "lasso": ("train", "caret", "lasso"),
}

responses = [
    "ss_noex",
    # "loess_noex",
    "dl",
    "ss_noex_rob_rew_1",
    # "loess_noex_rob_rew_1",
    "dl_rob_rew_1",
]


""" ########################################################################
Set up R - Environment - load data
"""


def _(str):
    return r["base"].get(str)


# packages
std_packages = ("stats", "utils", "datasets", "methods", "base")
ext_packages = ("earth", "mgcv", "randomForest", "glmnet", "lattice", "ggplot2")

"import / install packages --------------------------------------------------"
r = dict()  # object with R-pkg's
for pkg in std_packages:
    r[pkg] = rpackages.importr(pkg)
try:
    r["utils"].chooseCRANmirror(ind=38)  # Germany
except:
    print("Connection to CRAN mirrow failed")
packnames_to_install = [x for x in ext_packages if not rpackages.isinstalled(x)]
if len(packnames_to_install) > 0:
    print("install packages: ----")
    print(packnames_to_install)
    r["utils"].install_packages(StrVector(packnames_to_install))
for pkg in ext_packages:
    r[pkg] = rpackages.importr(pkg)

"load all ml_models --------------------------------------------------------"
methods = []
for response in responses:
    for i in methods_without_response.values():
        x, y, name = i
        for res in ["", "_res"]:
            methods.append((x, y, name + res, response))

# read: ml-models
ml_models = {}
for predict_method_suffix, package, name, response in methods:
    fname = f"ml_{predict_method_suffix}_{predict_method_suffix}_{name}_{package}_{response}.rds"
    fpathname = f"./data/computation_results/ml_models/R_small/{fname}"
    if os.path.exists(fpathname):
        try:
            obj = r["base"].readRDS(
                fpathname, refhook=_("eval")(_("parse")(text="function(x) NULL"))
            )
        except:
            obj = None
            print("could not load ml_model " + fname)
    else:
        print(f"file does not exist: {fpathname}")

    tpl = (predict_method_suffix, package, name, response)
    ml_models[tpl] = obj
ml_models
"loaded ----------------------------------------------------------------"


def get_R_df(df: pd.DataFrame):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    # turn scl_class into a factor
    scl_classes = _("$")(r_df, "scl_class")
    class_dict = _("seq")(2, 11)
    scl_classes = _("factor")(scl_classes, class_dict)
    r_df = _("$<-")(r_df, "scl_class", scl_classes)
    # r["utils"].str(r_df)
    return r_df


def correct_ndvi(df: pd.DataFrame, short_name: str, response: str):
    r_df = get_R_df(df)
    predict_method_suffix, package, name = methods_without_response[
        short_name.replace("_res", "")
    ]
    if "_res" in short_name:
        name = name + "_res"
    r_pred_fun = _(":::")(package, "predict." + predict_method_suffix)
    ml_model = ml_models[
        (predict_method_suffix, package, name, response.replace("ndvi_itpl_", ""))
    ]
    robj = r_pred_fun(ml_model, r_df)
    obj = np.asarray(robj)
    return np.squeeze(obj)
