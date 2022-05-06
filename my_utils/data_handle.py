import os
from re import S
import pandas as pd
import my_utils.pixel as pixel
import numpy as np
# %%


def csv_to_pickle(dir, update=False):
    """
    goes recursively through directory and for each
    csv file `file.csv` it adds the same file in the "pickel" 
    format as: `file.pkl`
    """
    dir_content = os.listdir(dir)
    dir_content = [os.path.join(dir, x) for x in dir_content]
    for element in dir_content:
        # element = os.path.join(directory, element)
        # print(element)
        if os.path.isdir(element):
            csv_to_pickle(element, update=update)
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


def get_pixels(frac, cloudy=False, train_test="train", WW_cereals="WW", 
    years=[2017, 2018, 2019, 2020, 2021], seed=None) -> list[pixel.Pixel]:
    """
    parameters
    ----------
    frac : fraction of items to use, frac="chance of item i to be included"
    dir : directory of .csv data
    train_test : possible values: "train", "test"
    WW_cereals : possible values: "WW", "cereals"
    years : list of year's to consider (integer)

    returns
    -------
    Intermediate result:
    # gives dictionary of year-dictionaries
    # each year dictionary consist of:
    #     "id" : numpy array with coordinates 
    #             (only a sample according to `frac`)
    #     "yie_path" : path to yield data
    #     "cov_path" : path to covariate data
    end result:pixel_list
    loading all train-cereals pixels (43275) takes ~100 sec
    """
    if seed is not None:
        np.random.seed(seed)
    if cloudy:
        dir = "data/yieldmapping_data/cloudy_data/yearly_train_test_sets"
    else:
        dir = "data/yieldmapping_data/yearly_train_test_sets"
    dir_content = os.listdir(dir)
    all_pixels = {}
    # if seed is not None:
    #     string = ""
    #     year_str = [string.append(str(year)[2:]) for year in years]
    #     file_name = "pixels"+str(frac)+str(cloudy)+train_test+WW_cereals+year_str+"_"+str(seed)+".pkl"
    #     file_path = "data/computation_results/scl/" + file_name + ".pkl"
    
    # # second try load object, or generate it if fail 
    # if os.path.isfile(file_path):
    #     with open(file_path, "rb") as f:
    #         result_df = pickle.load(f)
    # else:
    #     pixels = data_handle.get_pixels(frac,cloudy=True,WW_cereals=WW_cereals, years=years, seed=seed)
    #     result_df = calc_residuals(
    #         pixels, scl_class, interpol_method, **interpol_args)
    #     if save:
    #         with open(file_path, "wb") as f:
    #             pickle.dump(result_df, f)

##############################################
    for year in years:
        for file in dir_content:
            file = os.path.join(dir, file)
            file_base = os.path.basename(file)
            if (".csv" in file_base) \
                    and (str(year) in file_base) \
                    and (train_test in file_base) \
                    and (WW_cereals in file_base) \
                    and ("yield" in file_base):
                # print(file)
                df = read_df(file).sample(frac=frac)
                dirname = os.path.dirname(file)
                basename = os.path.basename(file)
                cov_path = os.path.join(
                    dirname, basename.replace("yield", "covariates"))
                all_pixels[str(year)] = {
                    "id": df.coord_id.to_numpy(),
                    "yie_path": file,
                    "cov_path": cov_path}
    # NOW: create list of pixels to return
    pixel_list = []
    failed_pixel_count = 0
    for year in all_pixels:
        d_cov = read_df(all_pixels[year]["cov_path"])
        d_yie = read_df(all_pixels[year]["yie_path"])
        for id in all_pixels[year]["id"]:
            try:
                pixel_list.append(pixel.Pixel(
                    d_cov, d_yie, coord_id=id, year=year))
            # except Exception as e:
            #     print(e)
            except:
                failed_pixel_count += 1
    print(f"{failed_pixel_count} pixels failed to generate")
    print(f"{len(pixel_list)} pixels have been generated")
    if seed is not None:
        print("help123")
    return pixel_list

# %%
