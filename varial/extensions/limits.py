"""
Limit derivation with theta: http://theta-framework.org
"""

import os
import ROOT
import math

import varial.tools
import varial.analysis
import theta_auto
import string
theta_auto.config.theta_dir = os.environ["CMSSW_BASE"] + "/theta"


class ThetaLimits(varial.tools.Tool):
    def __init__(
        self,
        input_path='../HistoLoader',
        asymptotic=True,
        dat_key=lambda w: w.is_data or w.is_pseudo_data,
        sig_key=lambda w: w.is_signal,
        bkg_key=lambda w: not any((w.is_signal, w.is_data, w.is_pseudo_data)),
        model_func=None,
        name=None,
    ):
        super(ThetaLimits, self).__init__(name)
        self.input_path = input_path
        self.asymptotic = asymptotic
        self.dat_key = dat_key
        self.sig_key = sig_key
        self.bkg_key = bkg_key
        self.model_func = model_func

    def _store_histos_for_theta(self, dat, sigs, bkgs):
        # create wrp
        wrp = varial.wrappers.Wrapper(name="ThetaHistos")
        if dat:
            setattr(wrp, 'histo__DATA', dat[0].histo)
        for bkg in bkgs:
            cat_name = bkg.in_file_path.split('/')[-2]
            setattr(wrp, cat_name + '__' + bkg.sample, bkg.histo)
        for sig in sigs:
            cat_name = sig.in_file_path.split('/')[-2]
            setattr(wrp, cat_name + '__' + sig.sample, sig.histo)

        # write manually
        filename = os.path.join(varial.analysis.cwd, wrp.name + ".root")
        f = ROOT.TFile.Open(filename, "RECREATE")
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
        self.model = self.model_func(self.cwd)

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
            res_exp=res_exp,
            res_obs=res_obs,
        )
        self.message(
            'INFO theta result: expected limit:\n' + str(self.result.res_exp))
        self.message(
            'INFO theta result: expected limit:\n' + str(self.result.res_obs))
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
            res_exp=self.result.res_exp,
            res_obs=self.result.res_obs,
            brs=self.brs
        )




class TriangleLimitPlots(varial.tools.Tool):
    def __init__(self,
        name=None
    ):
        super(TriangleLimitPlots, self).__init__(name)


    def run(self):
        parents = os.listdir(self.cwd+'/..')
        theta_tools = list(k for k in parents if k.startswith("ThetaLimit"))
        wrps = list(self.lookup_result('../' + k) for k in theta_tools)
        filename = os.path.join(varial.analysis.cwd, self.name + ".root")
        f = ROOT.TFile.Open(filename, "RECREATE")
        f.cd()
        tri_hist = ROOT.TH2F("triangular_limits", ";br to th;br to tz", 10, 0., 1., 10, 0., 1.)
        for w in wrps:
            br_th = float(w.brs['th'])
            br_tz = float(w.brs['tz'])
            # limit_f = float(w.res_exp.y[0])
            tri_hist.Fill(br_th, br_tz, w.res_exp.y[0])
        tri_hist.Write()
        f.Close()



