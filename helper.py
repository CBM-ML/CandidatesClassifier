import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from read_configs import *

import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from numpy import sqrt, log, argmax
import itertools
import treelite


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
    df_new = pd.DataFrame()

    non_log_x, log_x = read_log_vars(inp_file)



    for var in vars:
        if var in log_x:
            df_new['log('+ var+')'] = np.log(df[var])
        if var in non_log_x:
            df_new[var] = df[var]
    return df_new


def xgb_matr(x_train, y_train, x_test, y_test, cuts):
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

    dtrain = xgb.DMatrix(x_train[cuts], label = y_train)
    dtest=xgb.DMatrix(x_test[cuts], label = y_test)


    return dtrain, dtest


def AMS(y_true, y_predict, y_true1, y_predict1, output_path):
    roc_auc=roc_auc_score(y_true, y_predict)
    fpr, tpr, thresholds = roc_curve(y_true, y_predict,drop_intermediate=False ,pos_label=1)

    for i in range(len(thresholds)):
        if thresholds[i] > 1:
            thresholds[i]-=1

    S0 = sqrt(2 * ((tpr + fpr) * log((1 + tpr/fpr)) - tpr))
    S0 = S0[~np.isnan(S0)]
    S0 = S0[~np.isinf(S0)]
    xi = argmax(S0)
    S0_best_threshold = (thresholds[xi])

    roc_auc1=roc_auc_score(y_true1, y_predict1)
    fpr1, tpr1, thresholds1 = roc_curve(y_true1, y_predict1,drop_intermediate=False ,pos_label=1)

    for i in range(len(thresholds1)):
        if thresholds1[i] > 1:
            thresholds1[i]-=1

    S01 = sqrt(2 * ((tpr1 + fpr1) * log((1 + tpr1/fpr1)) - tpr1))
    S01 = S01[~np.isnan(S01)]
    S01 = S01[~np.isinf(S01)]
    xi1 = argmax(S01)
    S0_best_threshold1 = (thresholds[xi1])


    fig, ax = plt.subplots(figsize=(12, 8), dpi = 100)
    plt.plot(fpr, tpr, linewidth=3 ,linestyle=':',color='darkorange',label='ROC curve train (area = %0.6f)' % roc_auc)
    plt.plot(fpr1, tpr1, color='green',label='ROC curve test (area = %0.6f)' % roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random guess')
    #plt.scatter(fpr[xi], tpr[xi], marker='o', color='black', label= 'Best Threshold train set = '+"%.4f" % S0_best_threshold +'\n AMS = '+ "%.2f" % S0[xi])
    plt.scatter(fpr1[xi1], tpr1[xi1], marker='o', s=80, color='blue', label= 'Best Threshold test set = '+"%.4f" % S0_best_threshold1 +'\n AMS = '+ "%.2f" % S01[xi1])
    plt.xlabel('False Positive Rate', fontsize = 18)
    plt.ylabel('True Positive Rate', fontsize = 18)
    plt.legend(loc="lower right", fontsize = 18)
    plt.title('Receiver operating characteristic', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0, 1.02])
    #axs.axis([-0.01, 1, 0.9, 1])
    fig.tight_layout()
    fig.savefig(str(output_path)+'/hists.png')
    plt.show()

    roc_curve_data = dict()
    roc_curve_data["fpr_train"] = fpr
    roc_curve_data["tpr_train"] = tpr

    roc_curve_data["fpr_test"] = fpr1
    roc_curve_data["tpr_test"] = tpr1

    return S0_best_threshold, S0_best_threshold1, roc_curve_data


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize = 15)
    plt.xlabel('Predicted label',fontsize = 15)


def preds_prob(df, preds, true, dataset, output_path):
    if dataset =='train':
        label1 = 'XGB Predictions on the training data set'
    else:
        label1 = 'XGB Predictions on the test data set'
    fig, ax = plt.subplots(figsize=(12, 8))
    bins1=100
    plt.hist(df[preds], bins=bins1,facecolor='green',alpha = 0.3, label=label1)
    TP = df[(df[true]==1)]
    TN = df[(df[true]==0)]
    #TP[preds].plot.hist(ax=ax, bins=bins1,facecolor='blue', histtype='stepfilled',alpha = 0.3, label='True Positives/signal in predictions')
    hist, bins = np.histogram(TP[preds], bins=bins1)
    err = np.sqrt(hist)
    center = (bins[:-1] + bins[1:]) / 2


    hist1, bins1 = np.histogram(TN[preds], bins=bins1)
    err1 = np.sqrt(hist1)
    plt.errorbar(center, hist1, yerr=err1, fmt='o',
                 c='Red', label='Background in predictions')

    plt.errorbar(center, hist, yerr=err, fmt='o',
                 c='blue', label='Signal in predictions')

    ax.set_yscale('log')
    plt.xlabel('Probability',fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.legend(fontsize=18)
    ax.set_xticks(np.arange(0,1.1,0.1))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.show()
    fig.tight_layout()
    fig.savefig(str(output_path)+'/test_best_pred.png')


def diff_SB(df, signal_label):
    dfs = df[df[signal_label]==1]
    dfb = df[df[signal_label]==0]

    return dfs, dfb


def difference_df(df_orig, df_cut, cut):
    return pd.concat([df_orig[cut], df_cut[cut]]).drop_duplicates(keep=False)


def diff_SB_cut(df, target_label):
    dfs_cut = df[(df['xgb_preds1']==1) & (df[target_label]==1)]
    dfb_cut = df[(df['xgb_preds1']==1) & (df[target_label]==0)]

    return dfs_cut, dfb_cut



def save_model_lib(bst_model, output_path):
    bst = bst_model.get_booster()

    #create an object out of your model, bst in our case
    model = treelite.Model.from_xgboost(bst)
    #use GCC compiler
    toolchain = 'gcc'
    #parallel_comp can be changed upto as many processors as one have
    model.export_lib(toolchain=toolchain, libpath=output_path+'/xgb_model.so',
                     params={'parallel_comp': 4}, verbose=True)


    # Operating system of the target machine
    platform = 'unix'
    model.export_srcpkg(platform=platform, toolchain=toolchain,
                pkgpath=output_path+'/XGBmodel.zip', libname='xgb_model.so',
                verbose=True)
