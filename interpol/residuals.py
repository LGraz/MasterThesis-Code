#%% 
import os
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt


while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel
import my_utils.data_handle as data_handle
import my_utils.cv as cv

importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel
#%%
np.random.seed(4567)
pixels = data_handle.get_pixels(0.001, cloudy=True)

for pix in pixels:
    pix.get_smoothing_spline()

