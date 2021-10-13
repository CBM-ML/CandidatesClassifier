from hipe4ml.model_handler import ModelHandler
from dataclasses import dataclass
import xgboost as xgb

import matplotlib.pyplot as plt
from hipe4ml import plot_utils

import xgboost as xgb
import treelite


@dataclass
class XGBmodel():
    features_for_train: list
    hyper_pars_ranges: dict
    train_test_data: list
    output_path : str
    model_hdl: ModelHandler = (None, None, None)
    metrics: str = 'roc_auc'
    nfold: int = 3
    init_points: int = 1
    n_iter: int = 2
    n_jobs: int = -1




    def modelBO(self):
        model_clf = xgb.XGBClassifier()
        self.model_hdl = ModelHandler(model_clf, self.features_for_train)
        self.model_hdl.optimize_params_bayes(self.train_test_data, self.hyper_pars_ranges,
         self.metrics, self.nfold, self.init_points, self.n_iter, self.n_jobs)



    def train_test_pred(self):
        self.model_hdl.train_test_model(self.train_test_data)

        y_pred_train = self.model_hdl.predict(self.train_test_data[0], False)
        y_pred_test = self.model_hdl.predict(self.train_test_data[2], False)


        return y_pred_train, y_pred_test


    def save_predictions(self, filename):
        print(self.model_hdl.get_original_model())
        self.model_hdl.dump_original_model(self.output_path+'/'+filename, xgb_format=False)


    def load_model(self, filename):
        self.model_hdl.load_model_handler(filename)



    def save_model_lib(self):
        bst = self.model_hdl.model.get_booster()

        #create an object out of your model, bst in our case
        model = treelite.Model.from_xgboost(bst)
        #use GCC compiler
        toolchain = 'gcc'
        #parallel_comp can be changed upto as many processors as one have
        model.export_lib(toolchain=toolchain, libpath=self.output_path+'/xgb_model.so',
                         params={'parallel_comp': 4}, verbose=True)


        # Operating system of the target machine
        platform = 'unix'
        model.export_srcpkg(platform=platform, toolchain=toolchain,
                    pkgpath=self.output_path+'/XGBmodel.zip', libname='xgb_model.so',
                    verbose=True)



    def plot_dists(self):

        leg_labels = ['background', 'signal']
        ml_out_fig = plot_utils.plot_output_train_test(self.model_hdl, self.train_test_data, 100,
                                               False, leg_labels, True, density=True)

        plt.savefig(str(self.output_path)+'/thresholds.png')
