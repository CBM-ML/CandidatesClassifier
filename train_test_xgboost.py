import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from hipe4ml.plot_utils import plot_roc_train_test
from helper import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


class TrainTestXGBoost:
    def __init__(self, bst, dtrain, y_train, dtest, y_test):

        self.bst = bst
        self.dtrain = dtrain
        self.y_train = y_train
        self.dtest = dtest
        self.y_test = y_test

    def apply_predictions(self):

        bst_train= pd.DataFrame(data=self.bst.predict(self.dtrain, output_margin=False),  columns=["xgb_preds"])
        bst_train['issignal']=self.y_train


        bst_test= pd.DataFrame(data=self.bst.predict(self.dtest, output_margin=False),  columns=["xgb_preds"])
        bst_test['issignal']=self.y_test

        return bst_train, bst_test


    def features_importance(self, output_path):
        ax = xgb.plot_importance(self.bst)
        plt.rcParams['figure.figsize'] = [6, 3]
        ax.figure.tight_layout()
        ax.figure.savefig(str(output_path)+"/xgb_train_variables_rank.png")

    def save_ROC(self, y_pred_test, y_pred_train, output_path):

        leg_labels = ['background', 'signal']
        plot_roc_train_test(self.y_test, y_pred_test, self.y_train, y_pred_train, None, leg_labels)
        plt.tight_layout()
        plt.savefig(str(output_path)+"/roc_test_train.png")



    def CM_plot_train_test(self, x_train, best_train, x_test,best_test, output_path):
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
         plt.savefig(str(output_path)+'/confusion_matrix_extreme_gradient_boosting_train.png')

         cut_test = best_test
         x_test['xgb_preds1'] = ((x_test['xgb_preds']>cut_test)*1)
         cnf_matrix_test = confusion_matrix(x_test['issignal'], x_test['xgb_preds1'], labels=[1,0])
         np.set_printoptions(precision=2)
         fig_test, axs_test = plt.subplots(figsize=(10, 8))
         axs_test.yaxis.set_label_coords(-0.04,.5)
         axs_test.xaxis.set_label_coords(0.5,-.005)
         plot_confusion_matrix(cnf_matrix_test, classes=['signal','background'],
           title=' Test Dataset Confusion Matrix for XGB for cut > '+str(cut_test))
         plt.savefig(str(output_path)+'/confusion_matrix_extreme_gradient_boosting_test.png')
