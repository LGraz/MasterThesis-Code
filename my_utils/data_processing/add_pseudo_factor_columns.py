import numpy as np
import pandas as pd


def add_pseudo_factor_columns(df, colname, must_contain_labels=None):
    """if a matrix contains a factor-variable, add the factor-encoding-columns
    to the matrix

    Args:
        df (): DataFrame
        colname (str): column name
        must_contain_labels: list of labels which shall be there

    Returns:
        NewDataFrame, 
        list of new columnames
    """
    which_label, label_list = pd.factorize(df[colname])
    label_list_str = [colname + str(label) for label in label_list]
    n = len(which_label)
    if must_contain_labels is not None:
        for i in must_contain_labels:
            if i not in label_list_str:
                label_list_str.append(i)
    df_labels = pd.concat([pd.DataFrame({label: np.full(n, False)})
                           for label in label_list_str], axis=1)
    for label_int, label_str in zip(label_list, label_list_str):
        df_labels[label_str] = (which_label == np.where(
            label_list == label_int)[0][0])
    df_new = pd.concat(
        [df.reset_index(drop=True), df_labels.reset_index(drop=True)], axis=1)
    return df_new, label_list_str
