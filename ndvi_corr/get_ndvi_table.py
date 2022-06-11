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

for frac in [0.01, 0.1, 1]:
    frac = 0.01
    ndvi_table = get_ndvi_table.get_ndvi_table(frac, update=False)
    ndvi_table.to_pickle(
        "data/computation_results/ndvi_tables/ndvi_table_" + str(frac))
    ndvi_table
