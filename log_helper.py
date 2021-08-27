import numpy as np
import pandas as pd

from read_configs import *



def transform_df_to_log(df, vars, inp_file):
    """
    Transforms DataFrame to DataFrame with features in log scale
    Parameters
    ------------------------------------------------
    df: pandas.DataFrame
       input DataFrame

    vars: list of str
        vars to be transformed

    inp_file:
        config TOML file with list of features that should and shouldn't be
        transformed to log scale 
    """
    df_new = df.copy()

    non_log_x, log_x = read_log_vars(inp_file)


    for var in vars:
        if var in log_x:
            df_new[var+'_log'] = np.log(df_new[var])
            df_new = df_new.drop([var], axis=1)
    return df_new
