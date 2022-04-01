#%% 
from distutils.log import error
import pandas as pd
#%%
import numpy as np
import os
import warnings


if "my_utils" in os.getcwd():
    os.chdir("..")


""" Load 'unique_coord_ids' data frame """
try:
    unique_coord_ids = pd.read_pickle(r"data/computation_results/unique_coord_ids.pkl")
except:
    warnings.warn("\nData did not load. \n \
    Data will be generated using a very inefficient (O(n^2)) algorithm \n \
    around 28min of computation on my laptop")
    
    # Load orignal data
    WW = False  # only consider 'Winter Wheat' otherwise consider Cereals
    if WW:
        path_cov = os.path.join("data/yieldmapping_data", "WW_covariates_tot.csv")
    else:
        path_cov = os.path.join("data/yieldmapping_data",
                                "Cereals_covariates_tot.csv")
    d_cov = pd.read_csv(path_cov)
    
    # find indicies of unique coord_id's
    ind=[]
    coord_ids = d_cov.coord_id.unique()
    for i, id in enumerate(coord_ids):
        ind.append(d_cov.coord_id[id==d_cov.coord_id].index[0])
    
    # get rows&columns with interesting (and unique) information
    unique_coord_ids = d_cov.loc[ind, ['coord_id', 'FID', 'scene_id', 'product_uri',
           'x_coord', 'y_coord']]
    unique_coord_ids.to_pickle(r"data/computation_results/unique_coord_ids.pkl")


def get_xy_coord(id_string):
    x, y = map(int, map(float, id_string.split("_")))
    return x, y

def get_coord_id_within_dist(id_string, dist=40):
    """
    get a pandas series of coord_id's which are within the distance
    of 'dist' meters of the given id
    """
    x, y = get_xy_coord(id_string)
    return unique_coord_ids[
            np.sqrt((unique_coord_ids.x_coord-x)**2 + \
            (unique_coord_ids.y_coord-y)**2) <= dist
           ].coord_id

def is_alternating_coordinate(coord_id, rule=None):
    """
    rule :  'half' chess-like pixel picking
            picture:
                oxoxoxox
                xoxoxoxo
                oxoxoxox

            'quarter' only use a quarter of pixels with max mutual distance
            picture:
                oxoxoxoxoxoxo
                ooooooooooooo
                xoxoxoxoxoxox
                ooooooooooooo
                oxoxoxoxoxoxo
            
            'all' or `None`
                xxxxxxx
                xxxxxxx
    """
    x, y = get_xy_coord(coord_id)
    if (rule is None) | (rule == "all"):
        return True
    elif rule == "half":
        if (x%20) == 0: ## x='even' 
            return (y%20) == 10 ## True if y='odd'
        else:
            return (y%20) == 0 ## True if:  x='odd' & y='even'
    elif rule == "quarter":
        if (x%40) == 0: ## x='even'
            return (y%20) == 0
        elif (x%40) == 20: ## x='even'
            return (y%20) == 10
        else:
            return False
    else:
        raise Warning("this rule is not implemented")

def get_alternating_pixels_df(df, rule="all"):
    """
    takes DataFrame and deletes all rows which correspond to `False`
    according to the provided rule.

    for *rule* see doc of `is_alternating_coordinate`
    """
    fun = lambda x: is_alternating_coordinate(x,rule)
    return df[list(map(fun, df.coord_id))]

