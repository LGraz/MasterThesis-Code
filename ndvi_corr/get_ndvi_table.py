# %%
import os
import sys

while "ndvi" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

from my_utils.get_ndvi_table import get_ndvi_table

get_ndvi_table(1, update=True)
