import ROOT

from array import array

from cand_class.helper import *

from dataclasses import dataclass

@dataclass
class HistBuilder:

    output_path : str
    root_output_name = 'hists.root'

    def __post_init__(self):

        __hist_out = ROOT.TFile(self.output_path+'/'+self.root_output_name, "UPDATE")
        __hist_out.cd()

        ROOT.gDirectory.mkdir('Signal')
        ROOT.gDirectory.mkdir('Background')

        __hist_out.cd()
        __hist_out.cd('Signal')


        ROOT.gDirectory.mkdir('train')
        ROOT.gDirectory.mkdir('test')

        __hist_out.cd()
        ROOT.gDirectory.cd('Signal/train')
        ROOT.gDirectory.mkdir('pt_rap')
        ROOT.gDirectory.mkdir('hists')

        __hist_out.cd()
        ROOT.gDirectory.cd('Signal/test')
        ROOT.gDirectory.mkdir('pt_rap')
        ROOT.gDirectory.mkdir('hists')


        __hist_out.cd()
        __hist_out.cd('Background')

        ROOT.gDirectory.mkdir('train')
        ROOT.gDirectory.mkdir('test')


        __hist_out.cd()
        ROOT.gDirectory.cd('Background/train')
        ROOT.gDirectory.mkdir('pt_rap')
        ROOT.gDirectory.mkdir('hists')

        __hist_out.cd()
        ROOT.gDirectory.cd('Background/test')
        ROOT.gDirectory.mkdir('pt_rap')
        ROOT.gDirectory.mkdir('hists')

        __hist_out.Close()


    def roc_curve_root(self):
        __hist_out = ROOT.TFile(self.output_path+'/'+self.root_output_name, "UPDATE");

        __hist_out.cd()

        fpr = roc_curve_data['fpr_train']
        tpr = roc_curve_data['tpr_train']


        fpr1 = roc_curve_data['fpr_test']
        tpr1 = roc_curve_data['tpr_test']


        fpr_d_tr = array('d', fpr.tolist())
        tpr_d_tr = array('d', tpr.tolist())

        fpr_d_ts = array('d', fpr1.tolist())
        tpr_d_ts = array('d', tpr1.tolist())

        train_roc = ROOT.TGraph(len(fpr_d_tr), fpr_d_tr, tpr_d_tr)
        test_roc = ROOT.TGraph(len(fpr_d_ts), fpr_d_ts, tpr_d_ts)

        train_roc.SetLineColor(ROOT.kRed + 2)
        test_roc.SetLineColor(ROOT.kBlue + 2)


        train_roc.SetLineWidth(3)
        test_roc.SetLineWidth(3)


        train_roc.SetLineStyle(9)
        test_roc.SetLineStyle(9)


        train_roc.SetTitle("Receiver operating characteristic train")
        test_roc.SetTitle("Receiver operating characteristic test")

        train_roc.GetXaxis().SetTitle('FPR');
        train_roc.GetYaxis().SetTitle('TPR');

        test_roc.GetXaxis().SetTitle('FPR');
        test_roc.GetYaxis().SetTitle('TPR');

        train_roc.Write("Train_roc")
        test_roc.Write("Test_roc")

        __hist_out.Close()

    def pt_rap_root(self, df_orig, df_cut, difference, sign, x_range, y_range, data_name):

        if sign ==0:
            s_label = 'Background'
            m = 5

        if sign==1:
            s_label = 'Signal'
            m = 1

        rej = round((1 -  (df_cut.shape[0] / df_orig.shape[0])) * 100, 5)
        saved = round((df_cut.shape[0] / df_orig.shape[0]) * 100, 5)
        diff = df_orig.shape[0] - df_cut.shape[0]

        __hist_out = ROOT.TFile(self.output_path+'/'+self.root_output_name, "UPDATE");
        __hist_out.cd()
        ROOT.gDirectory.cd(s_label+'/'+data_name+'/'+'pt_rap')


        rapidity_orig = array('d', df_orig['rapidity'].values.tolist())
        pT_orig = array('d', df_orig['pT'].values.tolist())
        pT_rap_before_cut = ROOT.TH2D( 'pT_rap_before_ML'+data_name, 'pT_rap_before_ML_'+data_name, 100, min(x_range),
         max(x_range), 100, min(x_range), max(x_range))


        for i in range(len(rapidity_orig)):
            pT_rap_before_cut.Fill(rapidity_orig[i], pT_orig[i])

        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPalette(ROOT.kBird)

        pT_rap_before_cut.GetXaxis().SetTitle('rapidity');
        pT_rap_before_cut.GetYaxis().SetTitle('pT, GeV');
        pT_rap_before_cut.Draw('COLZ')





        rapidity_cut = array('d', df_cut['rapidity'].values.tolist())
        pT_cut = array('d', df_cut['pT'].values.tolist())

        pT_rap_cut = ROOT.TH2D( 'pT_rap_after_ML_'+data_name, 'pT_rap_after_ML_'+data_name, 100, min(x_range),
         max(x_range), 100, min(x_range), max(x_range))


        for i in range(len(rapidity_cut)):
            pT_rap_cut.Fill(rapidity_cut[i], pT_cut[i])

        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPalette(ROOT.kBird)

        pT_rap_cut.GetXaxis().SetTitle('rapidity');
        pT_rap_cut.GetYaxis().SetTitle('pT, GeV');
        pT_rap_cut.Draw('COLZ')





        rapidity_diff = array('d', difference['rapidity'].values.tolist())
        pT_diff = array('d', difference['pT'].values.tolist())

        pT_rap_diff = ROOT.TH2D('pT_rap_diff_'+data_name, 'pT_rap_diff_'+data_name, 100, min(x_range),
         max(x_range), 100, min(x_range), max(x_range))


        for i in range(len(rapidity_diff)):
            pT_rap_diff.Fill(rapidity_diff[i], pT_diff[i])

        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPalette(ROOT.kBird)

        pT_rap_diff.GetXaxis().SetTitle('rapidity');
        pT_rap_diff.GetYaxis().SetTitle('pT, GeV');
        pT_rap_diff.Draw('COLZ')


        pT_rap_before_cut.Write()
        pT_rap_cut.Write()
        pT_rap_diff.Write()

        __hist_out.Close()


    def hist_variables_root(self, mass_var, dfs_orig, dfb_orig, dfs_cut, dfb_cut,difference_s, sample):
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

        __hist_out = ROOT.TFile(self.output_path+'/'+self.root_output_name, "UPDATE");
        __hist_out.cd()


        for feature in dfs_orig.columns:

            dfs_orig_feat = array('d', dfs_orig[feature].values.tolist())
            dfb_orig_feat = array('d', dfb_orig[feature].values.tolist())


            dfs_cut_feat = array('d', dfs_cut[feature].values.tolist())
            dfb_cut_feat = array('d', dfb_cut[feature].values.tolist())


            dfs_diff_feat = array('d', difference_s[feature].values.tolist())


            dfs_orig_root = ROOT.TH1D('signal before ML '+feature, 'signal before ML '+feature, 500,
            min(dfs_orig[feature].values.tolist()), max(dfs_orig[feature].values.tolist()))

            for i in range(len(dfs_orig_feat)):
                dfs_orig_root.Fill(dfs_orig_feat[i])

            dfs_orig_root.GetXaxis().SetTitle(feature);

            dfs_orig_root.Draw()

            dfb_orig_root = ROOT.TH1D('background before ML '+feature, 'background before ML '+feature, 500,
            min(dfb_orig[feature].values.tolist()), max(dfb_orig[feature].values.tolist()))

            for i in range(len(dfb_orig_feat)):
                dfb_orig_root.Fill(dfb_orig_feat[i])

            dfb_orig_root.GetXaxis().SetTitle(feature);
            dfb_orig_root.Draw()


            dfs_cut_root = ROOT.TH1D('signal after ML '+feature, 'signal after ML '+feature, 500,
            min(dfs_cut[feature].values.tolist()), max(dfs_cut[feature].values.tolist()))

            for i in range(len(dfs_cut_feat)):
                dfs_cut_root.Fill(dfs_cut_feat[i])

            dfs_cut_root.GetXaxis().SetTitle(feature);
            dfs_cut_root.Draw()


            dfb_cut_root = ROOT.TH1D('background after ML '+feature, 'background after ML '+feature, 500,
            min(dfb_cut[feature].values.tolist()), max(dfb_cut[feature].values.tolist()))

            for i in range(len(dfb_cut_feat)):
                dfb_cut_root.Fill(dfb_cut_feat[i])

            dfb_cut_root.GetXaxis().SetTitle(feature);
            dfb_cut_root.Draw()


            dfs_diff_root = ROOT.TH1D('signal difference '+feature, 'signal difference '+feature, 500,
            min(difference_s[feature].values.tolist()), max(difference_s[feature].values.tolist()))

            for i in range(len(dfs_diff_feat)):
                dfs_diff_root.Fill(dfs_diff_feat[i])

            dfs_diff_root.GetXaxis().SetTitle(feature);
            dfs_diff_root.Draw()


            __hist_out.cd()

            __hist_out.cd('Signal'+'/'+sample+'/'+'hists')
            dfs_orig_root.Write()
            dfs_cut_root.Write()
            dfs_diff_root.Write()

            __hist_out.cd()

            __hist_out.cd('Background'+'/'+sample+'/'+'hists')

            dfb_orig_root.Write()
            dfb_cut_root.Write()

        __hist_out.Close()
