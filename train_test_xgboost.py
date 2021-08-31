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

        self.bst_train = None
        self.bst_test = None

    def apply_predictions(self):

        self.bst_train= pd.DataFrame(data=self.bst.predict(self.dtrain, output_margin=False),  columns=["xgb_preds"])
        self.bst_train['issignal']=self.y_train


        self.bst_test= pd.DataFrame(data=self.bst.predict(self.dtest, output_margin=False),  columns=["xgb_preds"])
        self.bst_test['issignal']=self.y_test

        return self.bst_train, self.bst_test


    def features_importance(self):
        ax = xgb.plot_importance(self.bst)
        plt.rcParams['figure.figsize'] = [6, 3]
        ax.figure.tight_layout()
        ax.figure.savefig(str(self.output_path)+"/xgb_train_variables_rank.png")



    def CM_plot_train_test(self, best_train, best_test):
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
         self.bst_train['xgb_preds1'] = ((self.bst_train['xgb_preds']>cut_train)*1)
         cnf_matrix_train = confusion_matrix(self.bst_train['issignal'], self.bst_train['xgb_preds1'], labels=[1,0])
         np.set_printoptions(precision=2)
         fig_train, axs_train = plt.subplots(figsize=(10, 8))
         axs_train.yaxis.set_label_coords(-0.04,.5)
         axs_train.xaxis.set_label_coords(0.5,-.005)
         plot_confusion_matrix(cnf_matrix_train, classes=['signal','background'],
          title=' Train Dataset Confusion Matrix for XGB for cut > '+str(cut_train))
         plt.savefig(str(self.output_path)+'/confusion_matrix_extreme_gradient_boosting_train.png')

         cut_test = best_test
         self.bst_test['xgb_preds1'] = ((self.bst_test['xgb_preds']>cut_test)*1)
         cnf_matrix_test = confusion_matrix(self.bst_test['issignal'], self.bst_test['xgb_preds1'], labels=[1,0])
         np.set_printoptions(precision=2)
         fig_test, axs_test = plt.subplots(figsize=(10, 8))
         axs_test.yaxis.set_label_coords(-0.04,.5)
         axs_test.xaxis.set_label_coords(0.5,-.005)
         plot_confusion_matrix(cnf_matrix_test, classes=['signal','background'],
           title=' Test Dataset Confusion Matrix for XGB for cut > '+str(cut_test))
         plt.savefig(str(self.output_path)+'/confusion_matrix_extreme_gradient_boosting_test.png')


    def preds_prob(self, preds, true, dataset):
        if dataset =='train':
            label1 = 'XGB Predictions on the training data set'
        else:
            label1 = 'XGB Predictions on the test data set'
        fig, ax = plt.subplots(figsize=(12, 8))
        bins1=100
        plt.hist(self.bst_test[preds], bins=bins1,facecolor='green',alpha = 0.3, label=label1)

        TP = self.bst_test[(self.bst_test[true]==1)]
        TN = self.bst_test[(self.bst_test[true]==0)]
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


    def pT_vs_rapidity(self, df_orig, df_cut, difference, sign, x_range, y_range, data_name):
        fig, axs = plt.subplots(1,3, figsize=(15, 4), gridspec_kw={'width_ratios': [1, 1, 1]})


        if sign ==0:
            s_label = 'Background'
            m = 5

        if sign==1:
            s_label = 'Signal'
            m = 1

        axs[0].set_aspect(aspect = 'auto')
        axs[1].set_aspect(aspect = 'auto')
        axs[2].set_aspect(aspect = 'auto')

        rej = round((1 -  (df_cut.shape[0] / df_orig.shape[0])) * 100, 5)
        diff = df_orig.shape[0] - df_cut.shape[0]
        axs[0].legend(shadow=True, title =str(len(df_orig))+' samples', fontsize =14)
        axs[1].legend(shadow=True, title =str(len(df_cut))+' samples', fontsize =14)
        axs[2].legend(shadow=True, title ='ML cut rejects \n'+ str(rej) +'% of '+ s_label +
        '\n ' + str(diff)+ ' samples were rejected ',
         fontsize =14)

        counts0, xedges0, yedges0, im0 = axs[0].hist2d(df_orig['rapidity'], df_orig['pT'] , range = [x_range, y_range], bins=100,
                    norm=mpl.colors.LogNorm(), cmap=plt.cm.rainbow)

        axs[0].set_title(s_label + ' candidates before ML cut '+data_name, fontsize = 16)
        axs[0].set_xlabel('rapidity', fontsize=15)
        axs[0].set_ylabel('pT, GeV', fontsize=15)


        mpl.pyplot.colorbar(im0, ax = axs[0])



        axs[0].xaxis.set_major_locator(MultipleLocator(1))
        axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))

        axs[0].xaxis.set_tick_params(which='both', width=2)


        fig.tight_layout()


        counts1, xedges1, yedges1, im1 = axs[1].hist2d(df_cut['rapidity'], df_cut['pT'] , range = [x_range, y_range], bins=100,
                    norm=mpl.colors.LogNorm(), cmap=plt.cm.rainbow)

        axs[1].set_title(s_label + ' candidates after ML cut '+data_name, fontsize = 16)
        axs[1].set_xlabel('rapidity', fontsize=15)
        axs[1].set_ylabel('pT, GeV', fontsize=15)

        mpl.pyplot.colorbar(im1, ax = axs[1])





        axs[1].xaxis.set_major_locator(MultipleLocator(1))
        axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))

        axs[1].xaxis.set_tick_params(which='both', width=2)

        fig.tight_layout()


        counts2, xedges2, yedges2, im2 = axs[2].hist2d(difference['rapidity'], difference['pT'] , range = [x_range, y_range], bins=100,
                    norm=mpl.colors.LogNorm(), cmap=plt.cm.rainbow)

        axs[2].set_title(s_label + ' difference ', fontsize = 16)
        axs[2].set_xlabel('rapidity', fontsize=15)
        axs[2].set_ylabel('pT, GeV', fontsize=15)

        mpl.pyplot.colorbar(im1, ax = axs[2])





        axs[2].xaxis.set_major_locator(MultipleLocator(1))
        axs[2].xaxis.set_major_formatter(FormatStrFormatter('%d'))

        axs[2].xaxis.set_tick_params(which='both', width=2)

        fig.tight_layout()

        fig.savefig(self.output_path+'/pT_rapidity_'+s_label+'_ML_cut_'+data_name+'.png')
