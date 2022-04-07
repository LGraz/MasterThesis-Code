import os
import pandas as pd


def to_pickle(directory, update=False):
    """
    goes recursively through directory and for each
    csv file `file.csv` it adds the same file in the "pickel" 
    format as: `file.pkl`
    """
    dir_content = os.listdir(directory)
    dir_content = [os.path.join(directory, x) for x in dir_content]
    for element in dir_content:
        # element = os.path.join(directory, element)
        print(element)
        if os.path.isdir(element):
            to_pickle(element, update=update)
        else:
            if ".csv" in element:
                pkl_name = element.replace(".csv", ".pkl")
                if (pkl_name not in dir_content) or update:
                    obj = pd.read_csv(element)
                    obj.to_pickle(pkl_name)
                    print("ADDED: " + pkl_name)


def read_df(file):
    """
    reads data frame and trys to do that by pickle
    """
    try:
        obj = pd.read_pickle(file.replace(".csv", ".pkl"))
    except:
        obj = pd.read_csv(file)
    return obj
