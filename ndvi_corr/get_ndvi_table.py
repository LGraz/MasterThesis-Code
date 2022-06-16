# %%
import os
import sys


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

# my librarys
import my_utils.get_ndvi_table as get_ndvi_table
################# END SETUP ###############################

for frac in [0.01]:
    frac = 0.01
    ndvi_table = get_ndvi_table.get_ndvi_table(
        frac, name="2017-20", update=False, get_pixels_kwargs={"years": [2017, 2018, 2019, 2020],
                                                               "cloudy": True, "WW_cereals": "cereals"})
    ndvi_table.to_pickle(
        "data/computation_results/ndvi_tables/ndvi_table_" + "2017-20_" + str(frac).replace(".", "") + ".pkl")
    ndvi_table

# %%
