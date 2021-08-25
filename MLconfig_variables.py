import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from matplotlib.font_manager import FontProperties

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import gc
import matplotlib as mpl
from scipy.stats import sem


def calculate_correlation(df, vars_to_corr, target_var) :
    """
    Calculates correlations with target variable variable and standart errors
    Parameters
    ------------------------------------------------
    df: pandas.DataFrame
        imput data
    vars_to_corr: list of str
         variables that correlate with target value
    target_var: str
          variable that correlates with another variables mentioned in list
    """


    mean = df[target_var].mean()
    sigma = df[target_var].std()

    correlation = []
    error = []

    for j in vars_to_corr :
        mean_j = df[j].mean()
        sigma_j = df[j].std()

        cov = (df[j] - mean_j) * (df[target_var] - mean) / (sigma*sigma_j)
        correlation.append(cov.mean())
        error.append(sem(cov))

    return correlation, error


def plot1Dcorrelation(vars_to_draw,var_to_corr, corr_signal, corr_signal_errors, corr_bg, corr_bg_errors, output_path):
    """
    Plots correlations
    Parameters
    ------------------------------------------------
    vars_to_draw: list of str
        variables that correlate with target value
    var_to_corr: str
         variables that correlate with target value
    corr_signal: list
          signal covariance coefficient between variable and target variable
    corr_signal_errors:
          signal covariance standart error of the mean
    corr_bg: list
          background covariance coefficient between variable and target variable
    corr_bg_errors:
         background covariance standart error of the mean
    output_path:
          path that contains output plot

    """

    fig, ax = plt.subplots(figsize=(20,10))
    plt.errorbar(vars_to_draw, corr_signal, yerr=corr_signal_errors, fmt='')
    plt.errorbar(vars_to_draw, corr_bg, yerr=corr_bg_errors, fmt='')
    ax.grid(zorder=0)
    ax.set_xticklabels(vars_to_draw, fontsize=25, rotation =70)
    ax.set_yticklabels([-0.5,-0.4,  -0.2,0, -0.2, 0.4], fontsize=25)
    plt.legend(('signal','background'), fontsize = 25)
    plt.title('Correlation of all variables with '+ var_to_corr+' along with SEM', fontsize = 25)
    plt.ylabel('Correlation coefficient', fontsize = 25)
    fig.tight_layout()
    fig.savefig(output_path+'/all_vars_corr-'+ var_to_corr+'.png')
