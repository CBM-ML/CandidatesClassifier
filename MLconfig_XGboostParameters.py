import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
import gc


class XGBoostParams:

    def __init__(self, params, dtrain):
        self.params = params
        self.dtrain = dtrain

    def bo_tune_xgb(self, max_depth, gamma, alpha, n_estimators ,learning_rate):

        params1 = {'max_depth': int(max_depth),
                  'gamma': gamma,
                  'alpha':alpha,
                  'n_estimators': n_estimators,
                  'learning_rate':learning_rate,
                  'subsample': 0.8,
                  'eta': 0.3,
                  'eval_metric': 'auc', 'nthread' : 7}

        cv_result = xgb.cv(params1, self.dtrain, num_boost_round=10, nfold=5)
        return  cv_result['test-auc-mean'].iloc[-1]

    def get_best_params(self):
        """
        Performs Bayesian Optimization and looks for the best parameters

        Parameters:
               None
        """
        #Invoking the Bayesian Optimizer with the specified parameters to tune
        xgb_bo = BayesianOptimization(self.bo_tune_xgb, {'max_depth': (4, 10),
                                                     'gamma': (0, 1),
                                                    'alpha': (2,20),
                                                     'learning_rate':(0,1),
                                                     'n_estimators':(100,500)
                                                    })
        #performing Bayesian optimization for 5 iterations with 8 steps of random exploration
        # with an #acquisition function of expected improvement
        xgb_bo.maximize(n_iter=1, init_points=1)

        max_param = xgb_bo.max['params']
        param1= {'alpha': max_param['alpha'], 'gamma': max_param['gamma'], 'learning_rate': max_param['learning_rate'],
         'max_depth': int(round(max_param['max_depth'],0)), 'n_estimators': int(round(max_param['n_estimators'],0)),
          'objective': 'reg:logistic'}
        gc.collect()


        #To train the algorithm using the parameters selected by bayesian optimization
        #Fit/train on training data
        bst = xgb.train(param1, self.dtrain)
        return bst
