"""
Limit derivation with theta: http://theta-framework.org
"""

from array import array
import collections
import cPickle
import numpy
import numpy.linalg as linalg
import ROOT
import glob
import os
import time
import pprint

import varial.generators as gen
import varial.sparseio
import varial.analysis
import varial.tools
import varial.pklio
import varial.monitor as monitor

import theta_auto
theta_auto.config.theta_dir = os.environ['CMSSW_BASE'] + '/../theta'


tex_table_mod_list = [
    ('_', r'\_'),        # escape underscore
    (r'\_{', '_{'),      # un-escape subscripts
    (r'\_', ' '),        # remove underscores
    ('(gauss)', ' '),
]


def tex_table_mod(table, mods=None):
    mods = mods or tex_table_mod_list
    for mod in mods:
        table = table.replace(*mod)
    return table


def add_th_curve(grps, th_x, th_y, legend='Theory', min_thy=None, max_thy=None):
    for g in grps:
        x_arr=array('f', th_x)
        y_arr=array('f', th_y)
        th_graph = ROOT.TGraph(len(x_arr), x_arr, y_arr)
        th_graph.SetLineStyle(2)
        # th_graph.SetLineColor(2)
        th_graph.SetLineWidth(2)
        min_log_y = min_thy or min(th_y)
        th_graph.GetXaxis().SetNdivisions(510, ROOT.kTRUE)

        th_wrp = varial.wrappers.GraphWrapper(
            th_graph,
            legend=legend,
            draw_option='C',
            val_y_min=min_log_y,
            val_y_max=max(th_y),
            is_th=True
        )
        g = list(g)
        g.append(th_wrp)
        res = varial.wrappers.WrapperWrapper(g)
        save_name = getattr(g[0], 'save_name', '')
        if save_name:
            res.save_name = save_name
        yield res

# def _write_postfit_histogram(histograms):
#     for reg in histograms:
#         hist_dict = histograms[reg]


######################################################### limit calculation ###
class ThetaLimitsBase(varial.tools.Tool):

    def __init__(
        self,
        model_func,
        asymptotic=True,
        calc_limits=True,
        limit_func=None,
        tex_table_mod_func=tex_table_mod,
        pvalue_func=None,
        postfit_func=lambda m, o: theta_auto.mle(m, options=o, input='data', n=1, with_covariance=True),
        get_postfit_hist='',
        hook_result_wrp=None,
        name=None,
    ):
        super(ThetaLimitsBase, self).__init__(name)
        self.model_func = model_func
        self.tex_table_mod = tex_table_mod_func
        self.model = None
        self.what = 'all'
        self.with_data = True
        self.with_signal = True
        self.pvalue_func = pvalue_func# or (lambda m: theta_auto.zvalue_approx(
                                      #                 m, input='data', n=1)
        self.postfit_func = postfit_func
        self.get_postfit_hist = get_postfit_hist
        if calc_limits:
            self.limit_func = limit_func or (
                theta_auto.asymptotic_cls_limits
                if asymptotic else
                lambda m: theta_auto.bayesian_limits(m, what=self.what)
            )
        else:
            self.limit_func = None
        self.hook_result_wrp = hook_result_wrp
        self.hist_name = ''

    def run_limits(self):

        # theta's config.workdir is broken for mle, where it writes temp data
        # into cwd. Therefore change into self.cwd.
        base_path = os.getcwd()
        os.chdir(self.cwd)
        res_exp, res_obs = {}, {}
        postfit = {}
        summary = {}
        p_vals = {}
        try:
            # setup theta
            theta_auto.config.workdir = '.'
            theta_auto.config.report = theta_auto.html_report('report.html')
            if not os.path.exists('plots'):
                os.mkdir('plots')

            options = theta_auto.Options()
            options.set('minimizer', 'strategy', 'robust')
            options.set('minimizer', 'minuit_tolerance_factor', '1000')

            # get model and let the fit run
            self.model = self.model_func(self.hist_name)

            if self.limit_func and self.with_signal:
                self.message('INFO calling theta limit func at %s' % self.cwd)
                res_exp, res_obs = self.limit_func(self.model)

            if self.pvalue_func:
                self.message('INFO calculating p-value')
                p_vals = self.pvalue_func(self.model)
                for z_dict in p_vals.itervalues():
                    z = z_dict['Z']
                    if isinstance(z, collections.Iterable):
                        z = numpy.median(z)
                    z_dict['p'] = theta_auto.Z_to_p(z)

            if self.postfit_func:
                self.message('INFO fetching post-fit parameters')
                if self.with_data:
                    try:
                        postfit = self.postfit_func(self.model, options)
                        sig_proc_dict = postfit.get(self.get_postfit_hist, None)
                        if sig_proc_dict:
                            parameter_values = {}
                            for p in self.model.get_parameters([]):
                                parameter_values[p] = sig_proc_dict[p][0][0]
                            histos = theta_auto.evaluate_prediction(self.model, parameter_values, include_signal=self.with_signal)
                            theta_auto.write_histograms_to_rootfile(histos, 'ThetaMLE_%s.root' % self.get_postfit_hist)
                    except RuntimeError as e:
                        self.message('WARNING error from theta: %s' % str(e.args))
            # shout it out loud            # theta_auto.write_histograms_to_rootfile(histos, 'histos-mle.root')

            summary = theta_auto.model_summary(self.model)
            try:
                theta_auto.config.report.write_html('result')
            except IOError as e:
                self.message('WARNING error from theta: %s' % str(e.args))
                

        except (IndexError, RuntimeError) as e:
            self.message('WARNING Error occured during theta run: %s' % e)

        finally:
            os.chdir(base_path)

        if summary:
            with open(self.cwd + 'rate_table.tex', 'w') as f:
                f.write(
                    self.tex_table_mod(
                        summary['rate_table'].tex()))

            for proc, table in summary['sysrate_tables'].iteritems():
                with open(self.cwd + 'sysrate_tables_%s.tex' % proc, 'w') as f:
                    f.write(
                        self.tex_table_mod(
                            table.tex()))

        self.result = varial.wrappers.Wrapper(
            name=self.name,
            # in order to access details, one must unpickle.
            res_exp=cPickle.dumps(res_exp),
            res_obs=cPickle.dumps(res_obs),
            summary=cPickle.dumps(summary),
            postfit_vals=cPickle.dumps(postfit),
            p_vals=p_vals,
            # mass_points=res_exp.x if res_exp else []
        )
        if self.hook_result_wrp:
            self.hook_result_wrp(self.result)


    def run(self):
        pass


######################################################### limit calculation ###
class ThetaLimits(ThetaLimitsBase):

    def __init__(
        self,
        model_func,
        input_path='../HistoLoader',
        input_path_sys='../HistoLoaderSys',
        filter_keyfunc=None,
        hook_loaded_histos=None,
        asymptotic=True,
        calc_limits=True,
        limit_func=None,
        cat_key=lambda _: 'histo',  # lambda w: w.category,
        dat_key=lambda w: w.is_data or w.is_pseudo_data,
        sig_key=lambda w: w.is_signal,
        bkg_key=lambda w: w.is_background,
        sys_key=None,
        tex_table_mod_func=tex_table_mod,
        pvalue_func=None,
        postfit_func=lambda m, o: theta_auto.mle(m, options=o, input='data', n=1, with_covariance=True),
        get_postfit_hist='',
        make_root_files_only=False,
        theta_root_file_name=None,
        hook_result_wrp=None,
        name=None,
    ):
        super(ThetaLimits, self).__init__(model_func, asymptotic,
            calc_limits, limit_func, tex_table_mod_func, pvalue_func,
            postfit_func, get_postfit_hist, hook_result_wrp, name)
        self.input_path = input_path
        self.input_path_sys = input_path_sys
        self.filter_keyfunc = filter_keyfunc
        self.hook_loaded_histos = hook_loaded_histos
        self.cat_key = cat_key
        self.dat_key = dat_key
        self.sig_key = sig_key
        self.bkg_key = bkg_key
        self.sys_key = sys_key
        self.make_root_files_only = make_root_files_only
        self.theta_root_file_name = theta_root_file_name

    def prepare_dat_sig_bkg(self, wrps):
        if self.filter_keyfunc:
            wrps = (w for w in wrps if self.filter_keyfunc(w))

        if self.hook_loaded_histos:
            wrps = self.hook_loaded_histos(wrps)

        wrps = list(wrps)
        dats = list(w for w in wrps if self.dat_key(w))
        sigs = list(w for w in wrps if self.sig_key(w))
        bkgs = list(w for w in wrps if self.bkg_key(w))
        return dats, sigs, bkgs

    def add_nominal_hists(self, wrp):

        if isinstance(self.input_path, str):
            self.input_path = [self.input_path]
        wrpwrps = []
        for i in self.input_path:
            if i.startswith('..'):
                i = os.path.join(self.cwd, i)
            wrpwrps += list(self.lookup_result(p) for p in glob.glob(i))
        wrps = []
        for ws in wrpwrps:
            if ws:
                wrps += list(w for w in ws)
        assert wrps, 'no input for path: %s' % self.input_path
        dats, sigs, bkgs = self.prepare_dat_sig_bkg(wrps)

        assert bkgs, 'no background histograms present.'
        # assert sigs, 'no signal histograms present.'

        if not sigs:
            self.message('WARNING No signal histograms, no limit setting possible.')
            # self.what = 'expected'
            self.with_signal = False
        if not dats:
            self.message('WARNING No data histogram, only expected limits.')
            self.what = 'expected'
            self.with_data = False

        for w in dats:
            setattr(wrp, self.cat_key(w) + '__DATA', w.histo)
        for w in bkgs:
            setattr(wrp, self.cat_key(w) + '__' + w.sample, w.histo)
        for w in sigs:
            setattr(wrp, self.cat_key(w) + '__' + w.sample, w.histo)

    def add_sys_hists(self, wrp):
        # wrps = self.lookup_result(self.input_path_sys)
        if isinstance(self.input_path_sys, str):
            self.input_path_sys = [self.input_path_sys]
        wrpwrps = []
        for i in self.input_path_sys:
            if i.startswith('..'):
                i = os.path.join(self.cwd, i)
            wrpwrps += list(self.lookup_result(p) for p in glob.glob(i))
        wrps = []
        for ws in wrpwrps:
            if ws:
                wrps += list(w for w in ws)
        if not wrps:
            return
        assert self.sys_key, 'sys_key is required. e.g. lambda w: w.sys_type'
        _, sigs, bkgs = self.prepare_dat_sig_bkg(wrps)

        def mk_name(w, sample=''):
            category = self.cat_key(w)
            sys = self.sys_key(w)
            return category + '__' + (sample or w.sample) + '__' + sys

        for w in bkgs:
            setattr(wrp, mk_name(w), w.histo)
        for w in sigs:
            setattr(wrp, mk_name(w), w.histo)

    @staticmethod
    def store_histos_for_theta(wrp):
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
        # clear workdir to make theta really run
        if glob.glob(self.cwd+'*'):
            os.system('rm -rf %s*' % self.cwd)

        # create wrp
        self.hist_name = self.theta_root_file_name+'.root' if self.theta_root_file_name else 'ThetaHistos%s.root' % hex(hash(time.clock()))
        wrp = varial.wrappers.Wrapper(
            name='ThetaHistos',
            file_path=os.path.join(self.cwd, self.hist_name),
        )

        # add histograms and store for theta
        self.add_nominal_hists(wrp)
        self.add_sys_hists(wrp)
        self.store_histos_for_theta(wrp)

        if not self.make_root_files_only:
            self.run_limits()
        # theta's config.workdir is broken for mle, where it writes temp data
        # into cwd. Therefore change into self.cwd.
        


######################################################### limit calculation ###
class ThetaLimitsFromFile(ThetaLimitsBase):

    def __init__(
        self,
        model_func,
        input_path=None,
        asymptotic=True,
        calc_limits=True,
        limit_func=None,
        tex_table_mod_func=tex_table_mod,
        pvalue_func=None,
        postfit_func=lambda m, o: theta_auto.mle(m, options=o, input='data', n=1, with_covariance=True),
        get_postfit_hist='',
        make_root_file=False,
        hook_result_wrp=None,
        name=None,
    ):
        super(ThetaLimitsFromFile, self).__init__(model_func, asymptotic,
            calc_limits, limit_func, tex_table_mod_func, pvalue_func,
            postfit_func, get_postfit_hist, hook_result_wrp, name)
        self.input_path = input_path
        self.make_root_file = make_root_file

    def run(self):
        # clear workdir to make theta really run
        if glob.glob(self.cwd+'*'):
            os.system('rm -rf %s*' % self.cwd)

        # create wrp
        self.hist_name = self.input_path

        if self.make_root_file:
            self.hist_name = 'ThetaHistos%s.root' % hex(hash(time.clock()))
            wrp = varial.wrappers.Wrapper(
                name='ThetaHistos',
                file_path=os.path.join(self.cwd, self.hist_name),
            )

            # add histograms and store for theta
            self.add_nominal_hists(wrp)
            self.add_sys_hists(wrp)
            self.store_histos_for_theta(wrp)

        self.run_limits()


################################################## plot nuisance parameters ###
class ThetaPostFitPlot(varial.tools.Tool):
    io = varial.pklio

    def __init__(
        self,
        input_path='../ThetaLimits',
        name=None,
    ):
        super(ThetaPostFitPlot, self).__init__(name)
        self.input_path = input_path

    @staticmethod
    def prepare_post_fit_items(post_fit_dict):
        post_fit_dict.pop('__chi2', None)
        post_fit_dict.pop('__ks', None)
        post_fit_dict.pop('__cov', None)
        post_fit_dict.pop('__nll', None)
        # pprint.pprint(post_fit_dict)
        return list(
            (name, val_err)
            for name, (val_err,) in sorted(post_fit_dict.iteritems())
            # if name not in ('__nll', '__cov', '__ks', '__chi2')
        )

    @staticmethod
    def prepare_pull_graph(n_items, post_fit_items):
        g = ROOT.TGraphAsymmErrors(n_items)
        for i, (_, (val, err)) in enumerate(post_fit_items):
            x, y = val, i + 1.5
            g.SetPoint(i, x, y)
            g.SetPointEXlow(i, err)
            g.SetPointEXhigh(i, err)

        g.SetLineStyle(1)
        g.SetLineWidth(1)
        g.SetLineColor(1)
        g.SetMarkerStyle(21)
        g.SetMarkerSize(1.25)
        return g

    @staticmethod
    def prepare_band_graphs(n_items):
        g68 = ROOT.TGraph(2*n_items+7)
        g95 = ROOT.TGraph(2*n_items+7)
        for a in xrange(0, n_items+3):
            g68.SetPoint(a, -1, a)
            g95.SetPoint(a, -2, a)
            g68.SetPoint(a+1+n_items+2, 1, n_items+2-a)
            g95.SetPoint(a+1+n_items+2, 2, n_items+2-a)
        g68.SetFillColor(ROOT.kGreen)
        g95.SetFillColor(ROOT.kYellow)
        return g68, g95

    @staticmethod
    def prepare_canvas(name, post_fit_items):
        c_name = 'cnv_post_fit_' + name
        c = ROOT.TCanvas(c_name, c_name, 720, len(post_fit_items)*38)
        c.SetTopMargin(0.06)
        c.SetRightMargin(0.02)
        c.SetBottomMargin(0.12)
        c.SetLeftMargin(0.35*700/650)
        c.SetTickx()
        c.SetTicky()
        return c

    @staticmethod
    def put_axis_foo(n_items, prim_graph, post_fit_items):
        prim_hist = prim_graph.GetHistogram()
        ax_1 = prim_hist.GetYaxis()
        ax_2 = prim_hist.GetXaxis()

        prim_graph.SetTitle('')
        ax_2.SetTitle('Post-fit values')
        ax_2.CenterTitle()
        #ax_2.SetTitle('deviation in units of std. dev.')
        ax_1.SetTitleSize(0.040)
        ax_2.SetTitleSize(0.050)
        ax_1.SetTitleOffset(1.4)
        ax_2.SetTitleOffset(1.0)
        ax_1.SetLabelSize(0.05)
        #ax_2.SetLabelSize(0.05)
        ax_1.SetRangeUser(0, n_items+2)
        ax_2.SetRangeUser(-4.4, 4.4)

        ax_1.Set(n_items+2, 0, n_items+2)
        ax_1.SetNdivisions(-414)
        for i, (uncert_name, _) in enumerate(post_fit_items):
            ax_1.SetBinLabel(i+2, varial.analysis.get_pretty_name(uncert_name))

    def mk_canvas(self, sig_name, post_fit_dict):
        n = len(post_fit_dict)
        items = self.prepare_post_fit_items(post_fit_dict)

        g = self.prepare_pull_graph(n, items)
        g68, g95 = self.prepare_band_graphs(n)
        cnv = self.prepare_canvas(sig_name, items)

        cnv.cd()
        g95.Draw('AF')
        g68.Draw('F')
        g.Draw('P')

        self.put_axis_foo(n, g95, items)
        g95.GetHistogram().Draw('axis,same')
        cnv.Modified()
        cnv.Update()

        return varial.wrp.CanvasWrapper(
            cnv, post_fit_items=items, pulls=g, g95=g95, g68=g68)

    def run(self):
        theta_res = self.lookup_result(self.input_path)
        postfit_vals = cPickle.loads(theta_res.postfit_vals)
        # pprint.pprint(postfit_vals)
        cnvs = (self.mk_canvas(sig, pfd)
                for sig, pfd in postfit_vals.iteritems())

        cnvs = varial.sparseio.bulk_write(cnvs, lambda c: c.name)
        self.result = list(cnvs)




################################################## plot correlation matrix ###

import numpy as np

class CorrelationMatrix(varial.tools.Tool):

    def __init__(
        self,
        input_path='../ThetaLimits',
        proc_name='',
        name=None,
    ):
        super(CorrelationMatrix, self).__init__(name)
        self.input_path = input_path
        self.proc_name = proc_name

    def run(self):
        theta_res = self.lookup_result(self.input_path)
        # pprint.pprint(theta_res)
        postfit_vals = cPickle.loads(theta_res.postfit_vals)
        if not postfit_vals.get(self.proc_name, None):
            self.message('WARNING process %s not in in postfit_vals' % self.proc_name)
            return
        if not postfit_vals[self.proc_name].get('__cov', None):
            self.message('WARNING no covariance matrix in postfit_vals')
            return

        # parameter_values = {}
        # for i in self.model.get_parameters([]):
        #     parameter_values[i] = postfit[''][i][0][0]

        # eval_pred = theta_auto.evaluate_prediction(self.model, parameter_values, include_signal = False)
        # pprint.pprint(eval_pred)
        theta_res = postfit_vals[self.proc_name]
        param_list = []
        for k, res in theta_res.iteritems():
            if any(k == i for i in ['__nll', '__cov', '__ks', '__chi2']):
                continue
            err_sq = res[0][1]*res[0][1]
            param_list.append((k, err_sq))


        cov_matrix = theta_res['__cov'][0]
        ind_dict = {}
        for i in xrange(cov_matrix.shape[0]):
            for ii in xrange(cov_matrix.shape[1]):
                entry = cov_matrix[i, ii]
                for proc, val in param_list:
                    if abs(val-entry) < 1e-6:
                        if i != ii:
                            self.message("WARNING row and column index don't match")
                        ind_dict[i] = proc
                if i not in ind_dict.keys():
                    ind_dict[i] = 'beta_signal'

        cov_matrix = np.matrix(cov_matrix)
        diag_matrix = np.matrix(np.sqrt(np.diag(np.diag(cov_matrix))))
        try:
            inv_matrix = diag_matrix.I
            corr_matrix = inv_matrix * cov_matrix * inv_matrix

            eigval, eigvec = linalg.eig(cov_matrix)

            corr_hist = ROOT.TH2F("correlation_matrix", "", len(param_list), 0, len(param_list), len(param_list), 0, len(param_list))
            cov_hist = ROOT.TH2F("covariance_matrix", "", len(param_list), 0, len(param_list), len(param_list), 0, len(param_list))
            eigval_hist = ROOT.TH2F("eigenvalue_matrix", "", len(param_list), 0, len(param_list), len(param_list), 0, len(param_list))
            eigvec_hist = ROOT.TH2F("eigenvec_matrix", "", len(param_list), 0, len(param_list), len(param_list), 0, len(param_list))
            for i in xrange(corr_matrix.shape[0]):
                if i not in ind_dict.keys():
                    continue
                corr_hist.GetXaxis().SetBinLabel(i+1, varial.analysis.get_pretty_name(ind_dict.get(i, 'unknown')))
                corr_hist.GetYaxis().SetBinLabel(i+1, varial.analysis.get_pretty_name(ind_dict.get(i, 'unknown')))
                cov_hist.GetXaxis().SetBinLabel(i+1, varial.analysis.get_pretty_name(ind_dict.get(i, 'unknown')))
                cov_hist.GetYaxis().SetBinLabel(i+1, varial.analysis.get_pretty_name(ind_dict.get(i, 'unknown')))
                eigvec_hist.GetXaxis().SetBinLabel(i+1, varial.analysis.get_pretty_name(ind_dict.get(i, 'unknown')))
                eigvec_hist.GetYaxis().SetBinLabel(i+1, varial.analysis.get_pretty_name(ind_dict.get(i, 'unknown')))
                eigval_hist.GetXaxis().SetBinLabel(i+1, varial.analysis.get_pretty_name(ind_dict.get(i, 'unknown'))+"'")
                eigval_hist.GetYaxis().SetBinLabel(i+1, varial.analysis.get_pretty_name(ind_dict.get(i, 'unknown'))+"'")
                for ii in xrange(corr_matrix.shape[1]):
                    entry_corr = corr_matrix[i,ii]
                    entry_cov = cov_matrix[i,ii]
                    entry_eigvec = eigvec[i,ii]
                    corr_hist.Fill(i, ii, entry_corr)
                    cov_hist.Fill(i, ii, entry_cov)
                    eigvec_hist.Fill(i, ii, entry_eigvec)
                    if i == ii:
                        entry_eigval = eigval[i]
                        eigval_hist.Fill(i, ii, entry_eigval)

            corr_hist.SetLabelSize(0.03, 'x')
            cov_hist.SetLabelSize(0.03, 'x')
            eigvec_hist.SetLabelSize(0.03, 'x')
            eigval_hist.SetLabelSize(0.03, 'x')
            corr_hist.SetMinimum(-1)
            corr_hist.SetMaximum(1)



            self.result = [varial.wrappers.HistoWrapper(corr_hist, save_name='corr_matrix'), 
                           varial.wrappers.HistoWrapper(cov_hist, save_name='cov_matrix'),
                           varial.wrappers.HistoWrapper(eigvec_hist, save_name='eigvec_matrix'), 
                           varial.wrappers.HistoWrapper(eigval_hist, save_name='eigval_matrix'),
                        ]
        except ValueError as e:
            self.message('WARNING no correlation matrix produced due to following error: %s' % e)
            self.result = None




        # cnvs = (self.mk_canvas(sig, pfd)
        #         for sig, pfd in theta_res.postfit_vals.iteritems())

        # cnvs = varial.sparseio.bulk_write(cnvs, lambda c: c.name)
        # self.result = list(cnvs)

######################################################### plot limit graphs ###
# class LimitGraphs(varial.tools.Tool):

#     def __init__(self,
#         limit_path='',
#         plot_obs=False,
#         plot_1sigmabands=False,
#         plot_2sigmabands=False,
#         axis_labels=('signal process', 'upper limit'),
#         name=None,
#     ):
#         super(LimitGraphs, self).__init__(name)
#         self.limit_path = limit_path
#         self.plot_obs = plot_obs
#         self.plot_1sigmabands = plot_1sigmabands
#         self.plot_2sigmabands = plot_2sigmabands
#         self.axis_labels = axis_labels

#     def prepare_sigma_band_graph(self, x_list, sig_low, sig_high):
#         n_items = len(x_list)
#         sig_graph = ROOT.TGraph(2*n_items)
#         for i in xrange(0, n_items):
#             sig_graph.SetPoint(i, x_list[i], sig_low[i])
#         for i in xrange(0, n_items):
#             sig_graph.SetPoint(i+n_items, x_list[n_items-i-1],
#                 sig_high[n_items-i-1])
#         return sig_graph

#     def make_sigma_band_graph(self, x_list, sigma_band_low, sigma_band_high, sigma_ind, selection=''):
#         assert type(sigma_ind) == int and (sigma_ind == 1 or sigma_ind == 2)
#         sigma_graph = self.prepare_sigma_band_graph(x_list, sigma_band_low,
#             sigma_band_high)
#         if sigma_ind == 1:
#             sigma_graph.SetFillColor(ROOT.kYellow)
#             legend='#pm 95% expected '+selection
#         else:
#             sigma_graph.SetFillColor(ROOT.kGreen)
#             legend='#pm 68% expected '+selection
#         sigma_graph.SetTitle(legend)
#         sigma_graph.GetXaxis().SetNdivisions(510, ROOT.kTRUE)

#         lim_wrapper = varial.wrappers.GraphWrapper(sigma_graph,
#             draw_option='F',
#             draw_option_legend='F',
#             val_y_min=min(sigma_band_low),
#             val_y_max=max(sigma_band_low)*10,
#             legend=legend,
#         )
#         return lim_wrapper

#     def make_graph(self, x_list, y_list, color, line_style, lim_type, selection=''):
#         x_arr = array('f', x_list)
#         y_arr = array('f', y_list)
#         lim_graph = ROOT.TGraph(len(x_arr), x_arr, y_arr)
#         lim_graph.SetLineColor(color)
#         lim_graph.SetLineWidth(2)
#         lim_graph.SetLineStyle(line_style)
#         lim_graph.GetXaxis().SetNdivisions(510, ROOT.kTRUE)
#         lim_wrapper = varial.wrappers.GraphWrapper(lim_graph,
#             legend=lim_type+' 95% CL '+selection,
#             draw_option='L',
#             val_y_min=min(y_list),
#             val_y_max=max(y_list)*10,
#         )
#         return lim_wrapper

#     def make_exp_graph(self, grp):
#         if len(grp) == 1:
#             wrp = grp[0]
#             theta_res_exp = cPickle.loads(wrp.res_exp)
#             x_list = theta_res_exp.x
#             y_list = theta_res_exp.y
#             color = varial.analysis.get_color(getattr(wrp, 'selection', '') or wrp.name, default=1)
#             selection = varial.settings.pretty_names.get(getattr(wrp, 'selection', '') or wrp.name, '')
#         elif len(grp) > 1:
#             x_list = []
#             y_list = []
#             selection = varial.settings.pretty_names.get(getattr(grp[0], 'selection', '') or grp[0].name, '')
#             color = varial.analysis.get_color(getattr(grp[0], 'selection', '') or grp[0].name, default=1)
#             wrps = grp.wrps
#             wrps = sorted(wrps, key=lambda w: w.mass_points[0] if w.mass_points else -1)
#             for wrp in wrps:
#                 if not wrp.mass_points:
#                     continue
#                 theta_res_exp = cPickle.loads(wrp.res_exp)
#                 x = theta_res_exp.x
#                 y = theta_res_exp.y
#                 assert len(x)==1 and len(y)==1, 'Not exactly one mass point in limit wrapper!'
#                 x_list.append(x[0])
#                 y_list.append(y[0])
#         lim_wrapper = self.make_graph(x_list, y_list, color, 3, 'expected', selection)
#         setattr(lim_wrapper, 'is_exp', True)
#         if hasattr(wrp, 'brs'):
#             setattr(lim_wrapper, 'save_name', 'tH%.0ftZ%.0fbW%.0f'\
#                 % (wrp.brs['h']*100, wrp.brs['z']*100, wrp.brs['w']*100))
#         return lim_wrapper

#     def make_obs_graph(self, grp):
#         if len(grp) == 1:
#             wrp = grp[0]
#             theta_res_obs = cPickle.loads(wrp.res_obs)
#             x_list = theta_res_obs.x
#             y_list = theta_res_obs.y
#             color = varial.analysis.get_color(getattr(wrp, 'selection', '') or wrp.name, default=1)
#             selection = varial.settings.pretty_names.get(getattr(wrp, 'selection', '') or wrp.name, '')
#         elif len(grp) > 1:
#             x_list = []
#             y_list = []
#             selection = varial.settings.pretty_names.get(getattr(grp[0], 'selection', '') or grp[0].name, '')
#             color = varial.analysis.get_color(getattr(grp[0], 'selection', '') or grp[0].name, default=1)
#             wrps = grp.wrps
#             wrps = sorted(wrps, key=lambda w: w.mass_points[0] if w.mass_points else -1)
#             for wrp in wrps:
#                 if not wrp.mass_points:
#                     continue
#                 theta_res_obs = cPickle.loads(wrp.res_obs)
#                 x = theta_res_obs.x
#                 y = theta_res_obs.y
#                 assert len(x)==1 and len(y)==1, 'Not exactly one mass point in limit wrapper!'
#                 x_list.append(x[0])
#                 y_list.append(y[0])
#         lim_wrapper = self.make_graph(x_list, y_list, color, 1, 'observed', selection)
#         setattr(lim_wrapper, 'is_obs', True)
#         if hasattr(wrp, 'brs'):
#             setattr(lim_wrapper, 'save_name', 'tH%.0ftZ%.0fbW%.0f'\
#                 % (wrp.brs['h']*100, wrp.brs['z']*100, wrp.brs['w']*100))
#         return lim_wrapper

#     def make_sigma_graph(self, grp, ind):
#         assert type(ind) == int and (ind == 1 or ind == 2)
#         if len(grp) == 1:
#             wrp = grp[0]
#             theta_res = cPickle.loads(wrp.res_exp)
#             x_list = theta_res.x
#             sigma_band_low = theta_res.bands[ind-1][0]
#             sigma_band_high = theta_res.bands[ind-1][1]
#             selection = varial.settings.pretty_names.get(getattr(wrp, 'selection', '') or wrp.name, '')
#         elif len(grp) > 1:
#             selection = varial.settings.pretty_names.get(getattr(grp[0], 'selection', '') or grp[0].name, '')
#             x_list = []
#             sigma_band_low = []
#             sigma_band_high = []
#             wrps = grp.wrps
#             wrps = sorted(wrps, key=lambda w: w.mass_points[0] if w.mass_points else -1)
#             for wrp in wrps:
#                 if not wrp.mass_points:
#                     continue
#                 theta_res = cPickle.loads(wrp.res_exp)
#                 x = theta_res.x
#                 sigma_low = theta_res.bands[ind-1][0]
#                 sigma_high = theta_res.bands[ind-1][1]
#                 assert len(x)==1 and len(sigma_low)==1 and len(sigma_high)==1, 'Not exactly one mass point in limit wrapper!'
#                 x_list.append(x[0])
#                 sigma_band_low.append(sigma_low[0])
#                 sigma_band_high.append(sigma_high[0])
#         lim_wrapper = self.make_sigma_band_graph(x_list, sigma_band_low, sigma_band_high, ind, selection)
#         if hasattr(wrp, 'brs'):
#             setattr(lim_wrapper, 'save_name', 'tH%.0ftZ%.0fbW%.0f'\
#                % (wrp.brs['h']*100, wrp.brs['z']*100, wrp.brs['w']*100))
#         return lim_wrapper

#     def set_draw_option(self, wrp, first=True):
#         if first:
#             wrp.draw_option+='A'
#         return wrp


#     def run(self):
#         if self.limit_path.startswith('..'):
#             theta_tools = glob.glob(os.path.join(self.cwd, self.limit_path))
#         else:
#             theta_tools = glob.glob(self.limit_path)
#         wrps = list(self.lookup_result(k) for k in theta_tools)
#         if any(not a for a in wrps):
#             wrps = gen.dir_content(self.limit_path, '*.info', 'result')
#         list_graphs=[]
#         wrps = sorted(wrps, key=lambda w: getattr(w, 'selection', ''))
#         wrps = gen.group(wrps, key_func=lambda w: getattr(w, 'selection', ''))
#         for w in wrps:
#             if self.plot_2sigmabands:
#                 list_graphs.append(self.set_draw_option(self.make_sigma_graph(w,
#                     1), not list_graphs))
#             if self.plot_1sigmabands:
#                 list_graphs.append(self.set_draw_option(self.make_sigma_graph(w,
#                     2), not list_graphs))
#             list_graphs.append(self.set_draw_option(self.make_exp_graph(w),
#                 not list_graphs))
#             if self.plot_obs:
#                 list_graphs.append(self.set_draw_option(self.make_obs_graph(w),
#                     not list_graphs))
#         for l in list_graphs:
#             x_title, y_title = self.axis_labels
#             l.obj.GetXaxis().SetTitle(x_title)
#             l.obj.GetYaxis().SetTitle(y_title)

#         self.result = varial.wrp.WrapperWrapper(list_graphs)


######################################################### plot limit graphs ###
class LimitGraphsNew(varial.tools.Tool):

    def __init__(self,
        limit_path='',
        hook_loaded_graphs=None,
        group_graphs=lambda ws: gen.group(ws, key_func=lambda w: ''),
        setup_graphs=None,
        split_mass=False,
        get_lim_params=None,
        plot_obs=False,
        plot_1sigmabands=False,
        plot_2sigmabands=False,
        axis_labels=('signal process', 'upper limit'),
        name=None,
    ):
        super(LimitGraphsNew, self).__init__(name)
        self.limit_path = limit_path
        self.hook_loaded_graphs = hook_loaded_graphs
        self.group_graphs = group_graphs
        self.setup_graphs = setup_graphs
        self.get_lim_params = self.get_lims_mass_split(plot_obs) if split_mass else self.get_lims_mass_comb(plot_obs)
        if get_lim_params:
            self.get_lim_params = get_lim_params
        self.plot_obs = plot_obs
        self.plot_1sigmabands = plot_1sigmabands
        self.plot_2sigmabands = plot_2sigmabands
        self.axis_labels = axis_labels

    @staticmethod
    def get_lims_mass_comb(plot_obs=True):
        def tmp(grp):
            wrp = grp[0]
            theta_res_exp = cPickle.loads(wrp.res_exp)
            if not theta_res_exp:
                self.message('ERROR Theta result empty.')
                raise RuntimeError  
            x_list = theta_res_exp.x
            y_exp_list = theta_res_exp.y
            if plot_obs:
                theta_res_obs = cPickle.loads(wrp.res_obs)
                y_obs_list = theta_res_obs.y
            else:
                y_obs_list = None
            sigma1_band_low = theta_res_exp.bands[1][0]
            sigma2_band_low = theta_res_exp.bands[0][0]
            sigma1_band_high = theta_res_exp.bands[1][1]
            sigma2_band_high = theta_res_exp.bands[0][1]
            return x_list, y_exp_list, y_obs_list, sigma1_band_low, sigma1_band_high, sigma2_band_low, sigma2_band_high
        return tmp

    @staticmethod
    def get_lims_mass_split(plot_obs=True):
        def tmp(grp):
            val_tup_list = []
            wrps = grp.wrps
            for wrp in wrps:
                theta_res_exp = cPickle.loads(wrp.res_exp)
                if not theta_res_exp:
                    continue
                x = theta_res_exp.x
                y_exp = theta_res_exp.y
                if plot_obs:
                    theta_res_obs = cPickle.loads(wrp.res_obs)
                    y_obs = theta_res_obs.y[0]
                else:
                    y_obs = None
                sigma1_low = theta_res_exp.bands[1][0]
                sigma2_low = theta_res_exp.bands[0][0]
                sigma1_high = theta_res_exp.bands[1][1]
                sigma2_high = theta_res_exp.bands[0][1]
                if not (len(x)==1 and len(sigma1_low)==1 and len(sigma1_high)==1 and len(sigma2_low)==1 and len(sigma2_high)==1):
                    monitor.message('limits.get_lims_mass_split', 'WARNING Not exactly one mass point in limit wrapper! ' +\
                        'Length of x/sigma1_low/sigma1_high/sigma2_low/sigma2_high: %s/%s/%s/%s/%s' % (str(len(x)), str(len(sigma1_low)), str(len(sigma1_high)), str(len(sigma2_low)), str(len(sigma2_high))))
                val_tup_list.append((x[0], y_exp[0], y_obs, sigma1_low[0], sigma2_low[0], sigma1_high[0], sigma2_high[0]))
            val_tup_list = sorted(val_tup_list, key=lambda w: w[0])
            x_list = list(w[0] for w in val_tup_list)
            y_exp_list = list(w[1] for w in val_tup_list)
            y_obs_list = list(w[2] for w in val_tup_list)
            sigma1_band_low = list(w[3] for w in val_tup_list)
            sigma2_band_low = list(w[4] for w in val_tup_list)
            sigma1_band_high = list(w[5] for w in val_tup_list)
            sigma2_band_high = list(w[6] for w in val_tup_list)
            return x_list, y_exp_list, y_obs_list, sigma1_band_low, sigma1_band_high, sigma2_band_low, sigma2_band_high
        return tmp

    def prepare_sigma_band_graph(self, x_list, sig_low, sig_high):
        n_items = len(x_list)
        sig_graph = ROOT.TGraph(2*n_items)
        for i in xrange(0, n_items):
            sig_graph.SetPoint(i, x_list[i], sig_low[i])
        for i in xrange(0, n_items):
            sig_graph.SetPoint(i+n_items, x_list[n_items-i-1],
                sig_high[n_items-i-1])
        return sig_graph

    def make_sigma_band_graph(self, x_list, sigma_band_low, sigma_band_high, sigma_ind, **kws):
        assert type(sigma_ind) == int and (sigma_ind == 1 or sigma_ind == 2)
        sigma_graph = self.prepare_sigma_band_graph(x_list, sigma_band_low,
            sigma_band_high)
        if sigma_ind == 1:
            sigma_graph.SetFillColor(ROOT.kYellow)
            legend='#pm 95% expected'
        else:
            sigma_graph.SetFillColor(ROOT.kGreen)
            legend='#pm 68% expected'
        sigma_graph.SetTitle(legend)
        sigma_graph.GetXaxis().SetNdivisions(510, ROOT.kTRUE)

        lim_wrapper = varial.wrappers.GraphWrapper(sigma_graph,
            draw_option='F',
            draw_option_legend='F',
            val_y_min=min(sigma_band_low),
            val_y_max=max(sigma_band_low)*10,
            legend=legend,
            save_name='lim_graph',
            file_path=self.cwd
        )
        lim_wrapper.__dict__.update(kws)
        return lim_wrapper

    def make_graph(self, x_list, y_list, color, line_style, lim_type, **kws):
        x_arr = array('f', x_list)
        y_arr = array('f', y_list)
        lim_graph = ROOT.TGraph(len(x_arr), x_arr, y_arr)
        # print color
        lim_graph.SetLineColor(color)
        lim_graph.SetLineWidth(2)
        lim_graph.SetLineStyle(line_style)
        lim_graph.GetXaxis().SetNdivisions(510, ROOT.kTRUE)
        lim_wrapper = varial.wrappers.GraphWrapper(lim_graph,
            legend=lim_type,
            draw_option='L',
            val_y_min=min(y_list),
            val_y_max=max(y_list)*10,
            save_name='lim_graph',
            file_path=self.cwd
        )
        lim_wrapper.__dict__.update(kws)
        return lim_wrapper

    def run(self):
        if self.limit_path.startswith('..'):
            theta_tools = glob.glob(os.path.join(self.cwd, self.limit_path))
        else:
            theta_tools = glob.glob(self.limit_path)
        wrps = list(self.lookup_result(k) for k in theta_tools)
        if any(not a for a in wrps):
            wrps = gen.dir_content(self.limit_path, '*.info', 'result')

        # wrps = list(wrps)
        # print len(wrps)

        if self.hook_loaded_graphs:
            wrps = self.hook_loaded_graphs(wrps)
            

        if self.group_graphs:
            wrps = self.group_graphs(wrps)
            

        if self.setup_graphs:
            wrps = self.setup_graphs(wrps)


        list_graphs=[]
        for w in wrps:
            x_list, y_exp_list, y_obs_list, sigma1_band_low, sigma1_band_high, sigma2_band_low, sigma2_band_high = self.get_lim_params(w)
            args = w.__dict__
            for i in ['wrps', 'name', 'type', 'klass', 'title']:
                args.pop(i, None)
            if self.plot_2sigmabands:
                list_graphs.append(self.make_sigma_band_graph(x_list, sigma2_band_low, sigma2_band_high, 1, **args))
            if self.plot_1sigmabands:
                list_graphs.append(self.make_sigma_band_graph(x_list, sigma1_band_low, sigma1_band_high, 2, **args))
            exp_col = args.pop('color', ROOT.kBlack)
            exp_style = args.pop('line_style', 3)
            list_graphs.append(self.make_graph(x_list, y_exp_list, exp_col, exp_style, 'Expected 95% CL', **args))
            if self.plot_obs and y_obs_list:
                list_graphs.append(self.make_graph(x_list, y_obs_list, ROOT.kBlack, 1, 'Observed 95% CL', **args))

        list_graphs[0].draw_option += 'A'

        for l in list_graphs:
            x_title, y_title = self.axis_labels
            l.obj.GetXaxis().SetTitle(x_title)
            l.obj.GetYaxis().SetTitle(y_title)

        self.result = varial.wrp.WrapperWrapper(list_graphs)




################################################## plot GOF tests ###
class ThetaGOFPlots(varial.tools.Tool):
    io = varial.pklio

    def __init__(
        self,
        input_path_data='../ThetaLimit',
        input_path_toys='../ThetaLimitToys',
        signal_name='',
        name=None,
    ):
        super(ThetaGOFPlots, self).__init__(name)
        self.input_path_data = input_path_data
        self.input_path_toys = input_path_toys
        self.signal_name = signal_name


    def run(self):
        theta_res_data = self.lookup_result(self.input_path_data)
        theta_res_toys = self.lookup_result(self.input_path_toys)
        postfit_vals_data = cPickle.loads(theta_res_data.postfit_vals)[self.signal_name]
        postfit_vals_toys = cPickle.loads(theta_res_toys.postfit_vals)[self.signal_name]
        ks_data = postfit_vals_data['__ks']
        chi2_data = postfit_vals_data['__chi2']
        ks_toys = postfit_vals_toys['__ks']
        chi2_toys = postfit_vals_toys['__chi2']

        # print ks_data
        # print ks_toys
        # print chi2_data
        # print chi2_toys
        ks_hist = ROOT.TH1F("toy_values_ks", "", int(max(ks_toys)*1.05), 0, int(max(ks_toys)*1.05))
        chi2_hist = ROOT.TH1F("toy_values_chi2", "", int(max(chi2_toys)*1.05), 0, int(max(chi2_toys)*1.05))

        for i in ks_toys:
            ks_hist.Fill(i)
        for i in chi2_toys:
            chi2_hist.Fill(i)

        # ks_graph.SetPoint(0, ks_data[0], 0)
        # ks_graph.SetPoint(1, ks_data[0], ks_hist.GetXaxis().GetXmax())
        # chi2_graph.SetPoint(0, chi2_data[0], 0)
        # chi2_graph.SetPoint(1, chi2_data[0], chi2_hist.GetXaxis().GetXmax())

        self.result = [varial.wrappers.HistoWrapper(ks_hist, legend='Toys', save_name='ks_hist', gof_data=ks_data[0], x_label='K.-S. Test Statistic'), 
                       varial.wrappers.HistoWrapper(chi2_hist, legend='Toys', save_name='chi2_hist', gof_data=chi2_data[0], x_label='#chi^{2} Test Statistic'),
                        ]
        # cnvs = (self.mk_canvas(sig, pfd)
        #         for sig, pfd in postfit_vals.iteritems())

        # cnvs = varial.sparseio.bulk_write(cnvs, lambda c: c.name)
        # self.result = list(cnvs)