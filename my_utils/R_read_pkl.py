import pandas as pd
import os
import pickle


def read_pickle_file(file):
    pickle_data = pd.read_pickle(file)
    return pickle_data


def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        return None
