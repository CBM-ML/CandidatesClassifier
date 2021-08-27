import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import sem
from scipy.stats import binned_statistic as b_s




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



def profile_plot_func(df,variable_xaxis, x_unit, variable_yaxis, sign, pdf_key, peak, edge_left, edge_right):

    if sign == 1:
        keyword = 'signal'
    if sign == 0:
        keyword = 'background'

    df = df[(df[variable_xaxis] < edge_right) & (df[variable_xaxis] > edge_left)]
    unit = r'mass, $ \frac{GeV}{c^2}$'



    fig, axs = plt.subplots(figsize=(20, 15))

    bin_means, bin_edges, binnumber = b_s(df[variable_xaxis],df[variable_yaxis], statistic='mean', bins=25)
    bin_std, bin_edges, binnumber = b_s(df[variable_xaxis],df[variable_yaxis], statistic='std', bins=25)
    bin_count, bin_edges, binnumber = b_s(df[variable_xaxis],df[variable_yaxis], statistic='count',bins= 25)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    nan_ind = np.where(np.isnan(bin_means))
    bin_centers = np.delete(bin_centers, nan_ind)
    bin_means = np.delete(bin_means, nan_ind)
    bin_count = np.delete(bin_count, nan_ind)
    bin_std = np.delete(bin_std , nan_ind)


    plt.errorbar(x=bin_centers, y=bin_means, yerr=(bin_std/np.sqrt(bin_count)), linestyle='none', marker='.',mfc='red', ms=10)


    plt.rcParams['font.size'] = '25'
    for label in(axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(25)


    plt.title('Mean of ' +variable_yaxis+ ' plotted versus bin centers of '+variable_xaxis+ \
              '('+keyword+')', fontsize=25)
    plt.xlabel(x_unit, fontsize=25)
    plt.ylabel("Mean of each bin with the SEM ($\dfrac{bin\ std}{\sqrt{bin\ count}}$) of bin", fontsize=25)


    plt.vlines(x=peak,ymin=bin_means.min(),ymax=bin_means.max(), color='r', linestyle='-')


    fig.tight_layout()
    plt.savefig(pdf_key,format='pdf')
