# %%
import os
import sys
sys.path.append(os.getcwd())

from my_utils.data_handle import csv_to_pickle

if __name__ == "__main__":
    # csv_to_pickle("./data/yieldmapping_data", update=False)
    csv_to_pickle(
        "./data/yieldmapping_data/cloudy_data/yearly_train_test_sets/", update=False)
    csv_to_pickle(
        "./data/yieldmapping_data/yearly_train_test_sets/", update=False)

    print("conversion from csv to pickel finished --------")
