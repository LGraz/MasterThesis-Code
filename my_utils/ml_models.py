from pandas.util import hash_pandas_object

import my_utils.data_handle as data_handle


def learn(model_class, df_train, response_name, covariates_name, name="", **model_args):
    """
    Description
    -----------
    Trains model and saves or loads it

    Parameters:
    ----------
    model_class : sklearn model (must have .fit method)

    df_train : DataFrame with "response_name" and "covariates_name"
    response_name : string with collumname
    covariates_name : list of strings with covariate-names
    name : string used to annotate filename
    """
    # prepare DataFrame (select collumns and remove na's)
    cols = covariates_name.copy()
    cols.append(response_name)
    df = df_train[cols]
    n_before = df.shape[0]
    df = df.dropna(axis=0, how='any')
    print(
        f"{n_before - df.shape[0]} of {n_before} rows removed because of missing values")

    # get hash
    my_hash = hash(tuple([
        hash(model_class),
        tuple(hash_pandas_object(df)),
        response_name,
        tuple(covariates_name),
        tuple(list(model_args.items()))
    ]))

    # filepath:
    filepath = "data/computation_results/ml_models/" + "model_" + \
        name + model_class.__name__ + "_" + response_name + "_" + str(my_hash)
    # try to load model
    model = data_handle.load(filepath)
    if model is not None:
        if model_class.__name__ not in type(model):
            print(
                f"loaded model of type:'{type(model)}' but model_class is {model_class.__name__}")
            raise Exception("failed to load correct model")
    else:
        # fit model
        X = df_train[covariates_name]
        y = df_train[response_name]
        model = model_class(**model_args)
        model.fit(X, y)
        # save model
        data_handle.save(model, filepath)
    return model
