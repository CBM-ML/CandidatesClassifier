import numpy as np
import pandas as pd

from read_configs import *

import xgboost as xgb



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


def xgb_matr(x_train, y_train, x_test, y_test):
    """
    To make machine learning algorithms more efficient on unseen data we divide
    our data into two sets. One set is for training the algorithm and the other
    is for testing the algorithm. If we don't do this then the algorithm can
    overfit and we will not capture the general trends in the data.

    Parameters
    ----------
    df_scaled: dataframe
          dataframe with mixed signal and background

    """

    dtrain = xgb.DMatrix(x_train, label = y_train)
    dtest=xgb.DMatrix(x_test, label = y_test)

    return dtrain, dtest
