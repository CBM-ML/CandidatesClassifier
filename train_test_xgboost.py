import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from hipe4ml.plot_utils import plot_roc_train_test
from helper import *
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


from matplotlib.font_manager import FontProperties

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import gc
import matplotlib as mpl



mpl.rc('figure', max_open_warning = 0)


class TrainTestXGBoost:
    def __init__(self, bst, dtrain, y_train, dtest, y_test, output_path):

        self.bst = bst
        self.dtrain = dtrain
        self.y_train = y_train
        self.dtest = dtest
        self.y_test = y_test
        self.output_path = output_path

    def apply_predictions(self):

        bst_train= pd.DataFrame(data=self.bst.predict(self.dtrain, output_margin=False),  columns=["xgb_preds"])
        bst_train['issignal']=self.y_train


        bst_test= pd.DataFrame(data=self.bst.predict(self.dtest, output_margin=False),  columns=["xgb_preds"])
        bst_test['issignal']=self.y_test

        return bst_train, bst_test


    def features_importance(self):
        ax = xgb.plot_importance(self.bst)
        plt.rcParams['figure.figsize'] = [6, 3]
        ax.figure.tight_layout()
        ax.figure.savefig(str(self.output_path)+"/xgb_train_variables_rank.png")



    def CM_plot_train_test(self, x_train, best_train, x_test,best_test):
         """
         Plots confusion matrix. A Confusion Matrix C is such that Cij is equal to
         the number of observations known to be in group i and predicted to be in
         group j. Thus in binary classification, the count of true positives is C00,
         false negatives C01,false positives is C10, and true neagtives is C11.

         Confusion matrix is applied to previously unseen by model data, so we can
         estimate model's performance

         Parameters
         ----------
         test_best: numpy.float32
                   best threshold

         x_train: dataframe
                 we want to get confusion matrix on training datasets
         """
         #lets take the best threshold and look at the confusion matrix
         cut_train = best_train
         x_train['xgb_preds1'] = ((x_train['xgb_preds']>cut_train)*1)
         cnf_matrix_train = confusion_matrix(x_train['issignal'], x_train['xgb_preds1'], labels=[1,0])
         np.set_printoptions(precision=2)
         fig_train, axs_train = plt.subplots(figsize=(10, 8))
         axs_train.yaxis.set_label_coords(-0.04,.5)
         axs_train.xaxis.set_label_coords(0.5,-.005)
         plot_confusion_matrix(cnf_matrix_train, classes=['signal','background'],
          title=' Train Dataset Confusion Matrix for XGB for cut > '+str(cut_train))
         plt.savefig(str(self.output_path)+'/confusion_matrix_extreme_gradient_boosting_train.png')

         cut_test = best_test
         x_test['xgb_preds1'] = ((x_test['xgb_preds']>cut_test)*1)
         cnf_matrix_test = confusion_matrix(x_test['issignal'], x_test['xgb_preds1'], labels=[1,0])
         np.set_printoptions(precision=2)
         fig_test, axs_test = plt.subplots(figsize=(10, 8))
         axs_test.yaxis.set_label_coords(-0.04,.5)
         axs_test.xaxis.set_label_coords(0.5,-.005)
         plot_confusion_matrix(cnf_matrix_test, classes=['signal','background'],
           title=' Test Dataset Confusion Matrix for XGB for cut > '+str(cut_test))
         plt.savefig(str(self.output_path)+'/confusion_matrix_extreme_gradient_boosting_test.png')


    def preds_prob(self, df, preds, true, dataset):
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
        fig.savefig(str(self.output_path)+'/test_best_pred.png')
