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
import ROOT

from array import array


mpl.rc('figure', max_open_warning = 0)


class TrainTestXGBoost:
    """
    Class used for train and test model on parameters found by Bayesian Optimization
    Parameters
    -------------------------------------------------
    bst:
        model to be used

    dtrain: xgboost Matrix
        train dataset

    y_train:
        train target label

    dtest: xgboost Matrix
        test dataset

    y_test:
        test target label

    bst_train:
        dataframe with predictions


    """


    def __init__(self, bst, dtrain, y_train, dtest, y_test, output_path):

        self.bst = bst
        self.dtrain = dtrain
        self.y_train = y_train
        self.dtest = dtest
        self.y_test = y_test
        self.output_path = output_path

        self.__bst_train = None
        self.__bst_test = None

        self.__train_best_thr = None
        self.__test_best_thr = None

        self.__train_pred = None
        self.__test_pred = None

        self.root_output_name = 'hists.root'

        hist_out = ROOT.TFile(self.output_path+'/'+self.root_output_name, "UPDATE");


        hist_out.cd()

        ROOT.gDirectory.mkdir('Signal')
        ROOT.gDirectory.mkdir('Background')

        hist_out.cd()
        hist_out.cd('Signal')


        ROOT.gDirectory.mkdir('train')
        ROOT.gDirectory.mkdir('test')

        hist_out.cd()
        ROOT.gDirectory.cd('Signal/train')
        ROOT.gDirectory.mkdir('pt_rap')
        ROOT.gDirectory.mkdir('roc')
        ROOT.gDirectory.mkdir('hists')

        hist_out.cd()
        ROOT.gDirectory.cd('Signal/test')
        ROOT.gDirectory.mkdir('pt_rap')
        ROOT.gDirectory.mkdir('roc')
        ROOT.gDirectory.mkdir('hists')


        hist_out.cd()
        hist_out.cd('Background')

        ROOT.gDirectory.mkdir('train')
        ROOT.gDirectory.mkdir('test')


        hist_out.cd()
        ROOT.gDirectory.cd('Background/train')
        ROOT.gDirectory.mkdir('pt_rap')
        ROOT.gDirectory.mkdir('roc')
        ROOT.gDirectory.mkdir('hists')

        hist_out.cd()
        ROOT.gDirectory.cd('Background/test')
        ROOT.gDirectory.mkdir('pt_rap')
        ROOT.gDirectory.mkdir('roc')
        ROOT.gDirectory.mkdir('hists')

        hist_out.Close()


    def apply_predictions(self):

        self.__bst_train= pd.DataFrame(data=self.bst.predict(self.dtrain, output_margin=False),  columns=["xgb_preds"])
        self.__bst_train['issignal']=self.y_train


        self.__bst_test= pd.DataFrame(data=self.bst.predict(self.dtest, output_margin=False),  columns=["xgb_preds"])
        self.__bst_test['issignal']=self.y_test

        return self.__bst_train, self.__bst_test


    def get_threshold(self, train_y, test_y):
        self.__train_best_thr, self.__test_best_thr = AMS(train_y, self.__bst_train['xgb_preds'], test_y, self.__bst_test['xgb_preds'], self.output_path)
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



    def features_importance(self):
        ax = xgb.plot_importance(self.bst)
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


    def preds_prob(self, preds, true, dataset):
        if dataset =='train':
            label1 = 'XGB Predictions on the training data set'
        else:
            label1 = 'XGB Predictions on the test data set'
        fig, ax = plt.subplots(figsize=(12, 8))
        bins1=100
        plt.hist(self.__bst_test[preds], bins=bins1,facecolor='green',alpha = 0.3, label=label1)

        TP = self.__bst_test[(self.__bst_test[true]==1)]
        TN = self.__bst_test[(self.__bst_test[true]==0)]
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




        hist_out = ROOT.TFile(self.output_path+'/'+self.root_output_name, "UPDATE");

        hist_out.cd()
        ROOT.gDirectory.cd(s_label+'/'+data_name+'/'+'pt_rap')


        rapidity_orig = array('d', df_orig['rapidity'].values.tolist())
        pT_orig = array('d', df_orig['pT'].values.tolist())

        pT_rap_before_cut = ROOT.TH2D( 'pT_rap_before_ML'+data_name, 'pT_rap_before_ML_'+data_name, 100, min(x_range),
         max(x_range), 100, min(x_range), max(x_range))


        for i in range(len(rapidity_orig)):
            pT_rap_before_cut.Fill(rapidity_orig[i], pT_orig[i])

        ROOT.gStyle.SetOptStat(0)


        ROOT.gStyle.SetPalette(ROOT.kBird)

        pT_rap_before_cut.Draw('COLZ')





        rapidity_cut = array('d', df_cut['rapidity'].values.tolist())
        pT_cut = array('d', df_cut['pT'].values.tolist())

        pT_rap_cut = ROOT.TH2D( 'pT_rap_after_ML_'+data_name, 'pT_rap_after_ML_'+data_name, 100, min(x_range),
         max(x_range), 100, min(x_range), max(x_range))


        for i in range(len(rapidity_cut)):
            pT_rap_cut.Fill(rapidity_cut[i], pT_cut[i])

        ROOT.gStyle.SetOptStat(0)


        ROOT.gStyle.SetPalette(ROOT.kBird)

        pT_rap_cut.Draw('COLZ')





        rapidity_diff = array('d', difference['rapidity'].values.tolist())
        pT_diff = array('d', difference['pT'].values.tolist())

        pT_rap_diff = ROOT.TH2D('pT_rap_diff_'+data_name, 'pT_rap_diff_'+data_name, 100, min(x_range),
         max(x_range), 100, min(x_range), max(x_range))


        for i in range(len(rapidity_diff)):
            pT_rap_diff.Fill(rapidity_diff[i], pT_diff[i])

        ROOT.gStyle.SetOptStat(0)


        ROOT.gStyle.SetPalette(ROOT.kBird)

        pT_rap_diff.Draw('COLZ')


        pT_rap_before_cut.Write()
        pT_rap_cut.Write()
        pT_rap_diff.Write()

        hist_out.Close()



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

            ax[0].set_title(str(feature) + ' MC '+ sample, fontsize = 25)
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

            ax[1].set_title(feature + ' MC '+ sample, fontsize = 25)
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

            ax[2].set_title(feature + ' MC '+ sample, fontsize = 25)
            ax[2].set_xlabel(feature, fontsize = 25)

            if feature!=mass_var:
                ax[2].set_yscale('log')

            fig.tight_layout()

            plt.savefig(pdf_key,format='pdf')
        pdf_key.close()
