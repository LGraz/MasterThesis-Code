import os
import sys
import importlib
import numpy as np
import scipy.optimize

while "interpolation" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.pixel as pixel
import my_utils.data_handle as data_handle
import my_utils.cv as cv

importlib.reload(data_handle)  # get changes in my_utils.pixel
importlib.reload(pixel)  # get changes in my_utils.pixel
importlib.reload(cv)  # get changes in my_utils.pixel


def main():
    frac = 0.0004
    np.random.seed(4321)
    pixels = data_handle.get_pixels(frac)
    return cv.get_res_list(
        pixels, 0.2, "smooth", {"method": "get_smoothing_spline", "k": np.inf}, {})


if __name__ == "__main__":
    result = main()
    for item in result:
        print(item)
