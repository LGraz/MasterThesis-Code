import os
import pandas as pd
import my_utils.pixel as pixel
import pickle
import numpy as np
# %%


def save(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        return None


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
                    df = pd.read_csv(element)
                    load(pkl_name)
                    print("ADDED: " + pkl_name)


def read_df(file):
    """
    reads data frame and trys to do that by pickle
    """
    try:
        load(file.replace(".csv", ".pkl"))
    except:
        print("No .pkl file for " + file)
        df = pd.read_csv(file)
    if isinstance(df, pd.DataFrame):
        if df.shape[0] == 0:
            raise Exception("zero rows dataframe")
    else:
        raise Exception(file + "is no DataFrame object")
    return df


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
    # try load pkl file
    if seed is not None:
        np.random.seed(seed)
        year_str = ""
        for year in years:
            year_str = year_str + str(year)[2:]
        file_name = "pixels" + str(frac) + str(cloudy) + train_test + \
            WW_cereals + year_str + "_" + str(seed) + ".pkl"
        file_path = "data/computation_results/pixels_pkl/" + file_name + ".pkl"
        # second try load object, or generate it if fail
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                pixels = pickle.load(f)
                if len(pixels) == 0:
                    raise Exception("no pixels where in: " + file_path)
                print(f"loaded {len(pixels)} pixels ---------------")
                return pixels

    if cloudy:
        dir = "data/yieldmapping_data/cloudy_data/yearly_train_test_sets"
    else:
        dir = "data/yieldmapping_data/yearly_train_test_sets"
    dir_content = os.listdir(dir)
    all_pixels = {}

    # Get (random) list of coord_id's
    # --> Trick: yield-datasets have one entry per coord_id
    found_any_csv = False
    for year in years:
        for file in dir_content:
            file = os.path.join(dir, file)
            file_base = os.path.basename(file)
            if (".csv" in file_base) \
                    and (str(year) in file_base) \
                    and (train_test in file_base) \
                    and (WW_cereals in file_base) \
                    and ("yield" in file_base):
                found_any_csv = True
                df = read_df(file).sample(frac=frac)
                dirname = os.path.dirname(file)
                basename = os.path.basename(file)
                cov_path = os.path.join(
                    dirname, basename.replace("yield", "covariates"))
                all_pixels[str(year)] = {
                    "id": df.coord_id.to_numpy(),
                    "yie_path": file,
                    "cov_path": cov_path}
    if not found_any_csv:  # try the same with pickle instead
        for year in years:
            for file in dir_content:
                file = os.path.join(dir, file)
                file_base = os.path.basename(file)
                # check for pkl this time
                if (".pkl" in file_base) \
                        and (str(year) in file_base) \
                        and (train_test in file_base) \
                        and (WW_cereals in file_base) \
                        and ("yield" in file_base):
                    found_any_csv = True
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
        with open(file_path, "wb") as f:
            pickle.dump(pixel_list, f)
    return pixel_list


def get_train_test_year(frac, year=2021, cloudy=True, WW_cereals="cereals", seed=4321) -> tuple[list[pixel.Pixel], list[pixel.Pixel]]:
    """
    Return
    ------
    (train, test)
    train : All pixels *except* year
    test : All pixels in year=year
    """
    test = get_pixels(frac=frac, train_test="test", cloudy=cloudy,
                      years=year, WW_cereals=WW_cereals, seed=seed)
    years = []
    for yea in [2017, 2018, 2019, 2020, 2021]:
        if year is not yea:
            years.append(yea)
    train = []
    for yea in years:
        train.extend(get_pixels(frac=frac, train_test="train", cloudy=cloudy,
                                years=yea, WW_cereals=WW_cereals, seed=seed))
    return (train, test)
    # %%
