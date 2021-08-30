import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


class TrainTestXGBoost:
    def __init__(self, bst, dtrain, y_train, dtest, y_test):

        self.bst = bst
        self.dtrain = dtrain
        self.y_train = y_train
        self.dtest = dtest
        self.y_test = y_test

    def apply_predictions(self, train_or_test):

        if train_or_test == 'train':

            bst_train= pd.DataFrame(data=self.bst.predict(self.dtrain, output_margin=False),  columns=["xgb_preds"])
            bst_train['issignal']=self.y_train

            return bst_train

        if train_or_test == 'test':

            bst_test= pd.DataFrame(data=self.bst.predict(self.dtest, output_margin=False),  columns=["xgb_preds"])
            bst_test['issignal']=self.y_test

            return bst_test


    def features_importance(self, output_path):
        ax = xgb.plot_importance(self.bst)
        plt.rcParams['figure.figsize'] = [6, 3]
        # plt.show()
        ax.figure.tight_layout()
        ax.figure.savefig(str(output_path)+"/xgb_train_variables_rank.png")
