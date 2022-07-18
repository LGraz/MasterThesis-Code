import os
import sys
import pickle
import copy
import importlib
import numpy as np
import pandas as pd

# change directory to code_dir
while True:
    changed = False
    for dir in ["my_utils", "interpol", "ndvi_corr"]:
        if ("code/" + dir) in os.getcwd():
            os.chdir("..")
            changed = True
    if not changed:
        break
sys.path.append(os.getcwd())

# reload
for module in sys.modules.values():
    importlib.reload(module)

# import (and reload) own librarys
import my_utils.data_handle as data_handle
import my_utils.pixel as pixel
import my_utils.pixel_multiprocess as pixel_multiprocess
import my_utils.strategies as strategies
import my_utils.itpl as itpl
import my_utils.plot_settings as plot_settings

import my_utils.cv as cv
import my_utils.coordinates as coordinates
import my_utils.loess as loess

import my_utils.scl_residuals as scl_residuals
import my_utils.get_ndvi_table as get_ndvi_table
import my_utils.ml_models as ml_models

print("Library's imported")
