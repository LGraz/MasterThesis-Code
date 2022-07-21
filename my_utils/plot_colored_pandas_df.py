import pandas as pd
import seaborn as sns


def write_df_to_latex(df, f_path_name):
    cm = sns.light_palette("black", as_cmap=True, reverse=False)
    df.style.background_gradient(axis=0)
    df_styled = df.apply(pd.to_numeric).style
    df_styled = df_styled.background_gradient(cmap=cm, axis=1).set_precision(3)

    # save
    text_file = open(f_path_name, "w")
    text_file.write(
        df_styled.to_latex(
            convert_css=True,
            #    clines="skip-last;data",
            multicol_align="|c|",
            hrules=True,
        )
    )
    text_file.close()
