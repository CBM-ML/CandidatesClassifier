import numpy as np
import pandas as pd

from read_configs import *



def transform_df_to_log(df, vars, inp_file):
    df_new = df.copy()

    non_log_x, log_x = read_log_vars(inp_file)


    for var in vars:
        if var in log_x:
            df_new[var+'_log'] = np.log(df_new[var])
            df_new = df_new.drop([var], axis=1)
    return df_new
