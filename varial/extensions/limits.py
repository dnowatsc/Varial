"""
Limit derivation with theta: http://theta-framework.org
"""

import os
import ROOT
import glob

import varial.tools
import varial.analysis
import varial.wrappers
import theta_auto
theta_auto.config.theta_dir = os.environ["CMSSW_BASE"] + "/theta"


class ThetaLimitWrapper(varial.wrappers.Wrapper):
    """
    Wrapper for theta 'plotdata' class.
    TODO: So far only stores x, y, xerrors and yerrors, check what else needs to be added!

    """
    def __init__(self, plotdata, **kws):
        super(ThetaLimitWrapper, self).__init__(**kws)
        self.x = None
        self.y = None
        self.xerrors = None
        self.yerrors = None
        self.x = plotdata.x
        self.y = plotdata.y
        self.xerrors = plotdata.xerrors
        self.yerrors = plotdata.yerrors

    def primary_object(self):
        return self.x


class ThetaLimits(varial.tools.Tool):
    def __init__(
        self,
        model_func,
        input_path='../HistoLoader',
        filter_keyfunc=None,
        asymptotic=True,
        cat_key=lambda _: 'histo',  # lambda w: w.category,
        dat_key=lambda w: w.is_data or w.is_pseudo_data,
        sig_key=lambda w: w.is_signal,
        bkg_key=lambda w: not any((w.is_signal, w.is_data, w.is_pseudo_data)),
        name=None,
    ):
        super(ThetaLimits, self).__init__(name)
        self.model_func = model_func
        self.input_path = input_path
        self.filter_keyfunc = filter_keyfunc
        self.asymptotic = asymptotic
        self.cat_key = cat_key
        self.dat_key = dat_key
        self.sig_key = sig_key
        self.bkg_key = bkg_key

    def _store_histos_for_theta(self, dats, sigs, bkgs, name="ThetaHistos"):
        # create wrp
        wrp = varial.wrappers.Wrapper(
            name=name,
            file_path=os.path.join(self.cwd, name + ".root"),
        )
        for w in dats:
            category = self.cat_key(w)
            setattr(wrp, category + '__DATA', w.histo)
        for w in bkgs:
            category = self.cat_key(w)
            setattr(wrp, category + '__' + w.sample, w.histo)
        for w in sigs:
            category = self.cat_key(w)
            setattr(wrp, category + '__' + w.sample, w.histo)

        # write manually
        f = ROOT.TFile.Open(wrp.file_path, "RECREATE")
        f.cd()
        for key, value in wrp.__dict__.iteritems():
            if isinstance(value, ROOT.TH1):
                value.SetName(key)
                value.Write()
        f.Close()
        return wrp

    def run(self):
        wrps = self.lookup_result(self.input_path)
        if not wrps:
            raise RuntimeError('No histograms present.')

        if self.filter_keyfunc:
            wrps = filter(self.filter_keyfunc, wrps)

        # print wrps
        dat = filter(self.dat_key, wrps)
        sig = filter(self.sig_key, wrps)
        bkg = filter(self.bkg_key, wrps)

        # do some basic input check
        if not bkg:
            raise RuntimeError('No background histograms present.')
        if not sig:
            raise RuntimeError('No signal histograms present.')
        if len(dat) > 1:
            raise RuntimeError('Too many data histograms present (>1).')
        if not dat:
            self.message('INFO No data histogram, only expected limits.')

        # setup theta
        theta_wrp = self._store_histos_for_theta(dat, sig, bkg)
        theta_auto.config.workdir = self.cwd
        theta_auto.config.report = theta_auto.html_report(os.path.join(
            self.cwd, 'report.html'
        ))
        plt_dir = os.path.join(self.cwd, 'plots')
        if not os.path.exists(plt_dir):
            os.mkdir(plt_dir)
        self.model = self.model_func(theta_wrp.file_path)

        # let the fit run
        options = theta_auto.Options()
        options.set('minimizer', 'strategy', 'robust')
        theta_auto.model_summary(self.model)
        if self.asymptotic:
            limit_func = lambda w: theta_auto.asymptotic_cls_limits(w)
        else: limit_func = lambda w: theta_auto.bayesian_limits(w, what='expected')
        res_exp, res_obs = limit_func(self.model)

        # shout it out loud
        self.result = varial.wrappers.Wrapper(
            name=self.name,
            res_exp_x=res_exp.x,  # TODO only TObjects or native python objects (list, dict, int, str ...) allowed
            res_exp_y=res_exp.y,
            res_exp_xerrors=res_exp.xerrors,
            res_exp_yerrors=res_exp.yerrors,
            # res_obs=varial.wrappers.ThetaLimitWrapper(res_obs),  # TODO only TObjects or native python objects (list, dict, int, str ...) allowed
        )
        # self.message(
        #     'INFO theta result: expected limit:\n' + str(self.result.res_exp))
        # self.message(
        #     'INFO theta result: observerd limit:\n' + str(self.result.res_obs))
        theta_auto.config.report.write_html(
            os.path.join(self.cwd, 'result'))

class ThetaLimitsBranchingRatios(ThetaLimits):
    def __init__(self,
        brs = None,
        *args,**kws
    ):
        super(ThetaLimitsBranchingRatios, self).__init__(*args, **kws)
        self.brs = brs

    def run(self):
        super(ThetaLimitsBranchingRatios, self).run()
        self.result = varial.wrappers.Wrapper(
            name=self.result.name,
            res_exp_x=self.result.res_exp_x,  # TODO only TObjects or native python objects (list, dict, int, str ...) allowed
            res_exp_y=self.result.res_exp_y,
            res_exp_xerrors=self.result.res_exp_xerrors,
            res_exp_yerrors=self.result.res_exp_yerrors,
            brs=self.brs
        )


class TriangleLimitPlots(varial.tools.Tool):
    def __init__(self,
        name=None,
        limit_rel_path=''
    ):
        super(TriangleLimitPlots, self).__init__(name)
        self.limit_rel_path = limit_rel_path


    def run(self):
        # parents = os.listdir(self.cwd+'/..')
        theta_tools = glob.glob(os.path.join(self.cwd+'..', self.limit_rel_path))
        wrps = list(self.lookup_result(k) for k in theta_tools)
        filename = os.path.join(varial.analysis.cwd, self.name + ".root")
        # f = ROOT.TFile.Open(filename, "RECREATE")
        # f.cd()
        tri_hist = ROOT.TH2F("triangular_limits", ";br to th;br to tz", 11, -0.05, 1.05, 11, -0.05, 1.05)
        for w in wrps:
            br_th = float(w.brs['th'])
            br_tz = float(w.brs['tz'])
            # limit_f = float(w.res_exp.y[0])
            tri_hist.Fill(br_th, br_tz, w.res_exp_y[0])
        # tri_hist.Write()
        self.result = [varial.wrappers.HistoWrapper(tri_hist, legend='twoD_plot')]
        # f.Close()






