import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from hipe4ml.plot_utils import plot_roc_train_test
from helper import *
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from dataclasses import dataclass

from matplotlib.font_manager import FontProperties

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import gc
import matplotlib as mpl
import ROOT

from array import array


mpl.rc('figure', max_open_warning = 0)


@dataclass
class ApplyXGB:

    y_pred_train : np.ndarray
    y_pred_test : np.ndarray

    y_train : np.ndarray
    y_test : np.ndarray

    output_path : str

    __bst_train : pd = pd.DataFrame()
    __bst_test : pd = pd.DataFrame()

    __train_best_thr : int = 0
    __test_best_thr : int = 0



    # root_output_name : str = 'hists.root'
    #
    # __hist_out : ROOT.TFile(output_path+'/'+root_output_name, "UPDATE")




    def apply_predictions(self):

        self.__bst_train["xgb_preds"] = self.y_pred_train
        self.__bst_train['issignal'] = self.y_train


        self.__bst_test["xgb_preds"] = self.y_pred_test
        self.__bst_test['issignal'] = self.y_test

        return self.__bst_train, self.__bst_test


    def get_threshold(self):

        self.__train_best_thr, self.__test_best_thr, roc_curve_data = AMS(self.y_train,
         self.__bst_train['xgb_preds'], self.y_test, self.__bst_test['xgb_preds'],
          self.output_path)


        return self.__train_best_thr, self.__test_best_thr


    def apply_threshold(self):
        cut_train = self.__train_best_thr
        self.__train_pred = ((self.__bst_train['xgb_preds']>cut_train)*1)

        cut_test = self.__test_best_thr
        self.__test_pred = ((self.__bst_test['xgb_preds']>cut_test)*1)

        return self.__train_pred, self.__test_pred


    def get_result(self, x_train, x_test):

        train_with_preds = x_train.copy()
        train_with_preds['xgb_preds1'] = self.__train_pred.values


        test_with_preds = x_test.copy()
        test_with_preds['xgb_preds1'] = self.__test_pred.values

        return train_with_preds, test_with_preds


    def features_importance(self, bst):
        # this one needs to be tested
        ax = xgb.plot_importance(bst)
        plt.rcParams['figure.figsize'] = [6, 3]
        ax.figure.tight_layout()
        ax.figure.savefig(str(self.output_path)+"/xgb_train_variables_rank.png")


    def CM_plot_train_test(self):
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

         cnf_matrix_train = confusion_matrix(self.__bst_train['issignal'], self.__train_pred, labels=[1,0])
         np.set_printoptions(precision=2)
         fig_train, axs_train = plt.subplots(figsize=(10, 8))
         axs_train.yaxis.set_label_coords(-0.04,.5)
         axs_train.xaxis.set_label_coords(0.5,-.005)
         plot_confusion_matrix(cnf_matrix_train, classes=['signal','background'],
          title=' Train Dataset Confusion Matrix for XGB for cut > '+str(self.__train_best_thr))
         plt.savefig(str(self.output_path)+'/confusion_matrix_extreme_gradient_boosting_train.png')


         cnf_matrix_test = confusion_matrix(self.__bst_test['issignal'], self.__test_pred, labels=[1,0])
         np.set_printoptions(precision=2)
         fig_test, axs_test = plt.subplots(figsize=(10, 8))
         axs_test.yaxis.set_label_coords(-0.04,.5)
         axs_test.xaxis.set_label_coords(0.5,-.005)
         plot_confusion_matrix(cnf_matrix_test, classes=['signal','background'],
           title=' Test Dataset Confusion Matrix for XGB for cut > '+str(self.__test_best_thr))
         plt.savefig(str(self.output_path)+'/confusion_matrix_extreme_gradient_boosting_test.png')



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
        saved = round((df_cut.shape[0] / df_orig.shape[0]) * 100, 5)
        diff = df_orig.shape[0] - df_cut.shape[0]
        axs[0].legend(shadow=True, title =str(len(df_orig))+' samples', fontsize =14)
        axs[1].legend(shadow=True, title =str(len(df_cut))+' samples', fontsize =14)
        axs[2].legend(shadow=True, title ='ML cut saves \n'+ str(saved) +'% of '+ s_label, fontsize =14)



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



    def hist_variables(self, mass_var, dfs_orig, dfb_orig, dfs_cut, dfb_cut, difference_s, sample, pdf_key):
        """
        Applied quality cuts and created distributions for all the features in pdf
        file
        Parameters
        ----------
        df_s: dataframe
              signal
        df_b: dataframe
              background
        feature: str
                name of the feature to be plotted
        pdf_key: PdfPages object
                name of pdf document with distributions
        """

        # hist_out = ROOT.TFile(self.output_path+'/'+self.root_output_name, "UPDATE");
        # hist_out.cd()


        for feature in dfs_orig.columns:
            fig, ax = plt.subplots(3, figsize=(20, 10))


            fontP = FontProperties()
            fontP.set_size('xx-large')

            ax[0].hist(dfs_orig[feature], label = 'signal', bins = 500, alpha = 0.4, color = 'blue')
            ax[0].hist(dfb_orig[feature], label = 'background', bins = 500, alpha = 0.4, color = 'red')
            ax[0].legend(shadow=True,title = 'S/B='+ str(round(len(dfs_orig)/len(dfb_orig), 3)) +

                       '\n S samples:  '+str(dfs_orig.shape[0]) + '\n B samples: '+ str(dfb_orig.shape[0]) +
                       '\nquality cuts ',
                       title_fontsize=15, fontsize =15, bbox_to_anchor=(1.05, 1),
                        loc='upper left', prop=fontP,)

            ax[0].set_xlim(dfb_orig[feature].min(), dfb_orig[feature].max())

            ax[0].xaxis.set_tick_params(labelsize=15)
            ax[0].yaxis.set_tick_params(labelsize=15)

            ax[0].set_title(str(feature) + ' MC '+ sample + ' before ML cut', fontsize = 25)
            ax[0].set_xlabel(feature, fontsize = 25)

            if feature!=mass_var:
                ax[0].set_yscale('log')

            fig.tight_layout()


            ax[1].hist(dfs_cut[feature], label = 'signal', bins = 500, alpha = 0.4, color = 'blue')
            ax[1].hist(dfb_cut[feature], label = 'background', bins = 500, alpha = 0.4, color = 'red')
            ax[1].legend(shadow=True,title = 'S/B='+ str(round(len(dfs_cut)/len(dfb_cut), 3)) +
                       '\n S samples:  '+str(dfs_cut.shape[0]) + '\n B samples: '+ str(dfb_cut.shape[0]) +
                       '\nquality cuts + ML cut',
                        title_fontsize=15, fontsize =15, bbox_to_anchor=(1.05, 1),
                        loc='upper left', prop=fontP,)


            ax[1].set_xlim(dfb_orig[feature].min(), dfb_orig[feature].max())

            ax[1].xaxis.set_tick_params(labelsize=15)
            ax[1].yaxis.set_tick_params(labelsize=15)

            ax[1].set_title(feature + ' MC '+ sample+ ' after ML cut', fontsize = 25)
            ax[1].set_xlabel(feature, fontsize = 25)

            if feature!='mass':
                ax[1].set_yscale('log')

            fig.tight_layout()




            ax[2].hist(difference_s[feature], label = 'signal', bins = 500, alpha = 0.4, color = 'blue')
            ax[2].legend(shadow=True,title ='S samples: '+str(len(difference_s)) +'\nsignal difference',
                        title_fontsize=15, fontsize =15, bbox_to_anchor=(1.05, 1),
                        loc='upper left', prop=fontP,)


            ax[2].set_xlim(dfb_orig[feature].min(), dfb_orig[feature].max())

            ax[2].xaxis.set_tick_params(labelsize=15)
            ax[2].yaxis.set_tick_params(labelsize=15)

            ax[2].set_title(feature + ' MC '+ sample +' signal difference', fontsize = 25)
            ax[2].set_xlabel(feature, fontsize = 25)

            if feature!=mass_var:
                ax[2].set_yscale('log')

            fig.tight_layout()

            plt.savefig(pdf_key,format='pdf')
        pdf_key.close()
