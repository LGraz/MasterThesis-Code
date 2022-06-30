"""Generate a colored table which compares several itpl-methods
    using the out-of-bag residuals and applying several statistics to them

    Requires: calculated ndvi table
    """
# %%
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
while "interpol" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())

# get data
ndvi_table = pd.read_pickle(
    "./data/computation_results/ndvi_tables/ndvi_table_2017-20_0001.pkl")
ndvi_table = ndvi_table.loc[(ndvi_table.scl_class == 4) | (
    ndvi_table.scl_class == 5)]

# choose desired columnames
blacklist = [
    # "fourier_rob"
]
colnames = [i for i in ndvi_table.columns if (
    "ndvi_itpl" in i) and not any(string in i for string in blacklist)]


def rename_colnames(x: str):
    x = x.replace("ndvi_itpl_", "")
    x = x.replace("_rew_1", "")
    x = x.replace("_noex", "")
    x = x.replace("_", " ")
    return x


# define statistics to apply later
statistic_dict = {
    "rmse": lambda res: np.sqrt(np.mean(np.square(res))),
    "qtile50": lambda res: np.quantile(np.abs(res), 0.50),
    "qtile75": lambda res: np.quantile(np.abs(res), 0.75),
    "qtile85": lambda res: np.quantile(np.abs(res), 0.85),
    "qtile90": lambda res: np.quantile(np.abs(res), 0.90),
    "qtile95": lambda res: np.quantile(np.abs(res), 0.95),
}

# apply statistic to OOB-residuals and write to dataframe
df = pd.DataFrame(columns=list(map(rename_colnames, colnames)),
                  index=statistic_dict.keys())
for colname in colnames:
    res = ndvi_table[colname].to_numpy(
    ) - ndvi_table["ndvi_observed"].to_numpy()
    for key, statistic in statistic_dict.items():
        df.at[key, rename_colnames(colname)] = np.round(statistic(res), 3)

# color dataframe
cm = sns.light_palette("black", as_cmap=True, reverse=False)
df.style.background_gradient(axis=0)
df_styled = df.apply(pd.to_numeric).style.background_gradient(
    cmap=cm, axis=1).set_precision(3)
df_styled
# %%
text_file = open(
    "../latex/tex/chapters/misc/table_cv-statistics_itpl-methods.tex", "w")
text_file.write(df_styled.to_latex(convert_css=True))
text_file.close()

# %%
