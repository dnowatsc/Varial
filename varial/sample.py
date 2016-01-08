import inspect
import itertools
import collections
import os.path
import glob

import settings
import wrappers
import monitor

# TODO: move Sample class to extensions.cmsrun??


class Sample(wrappers.WrapperBase):
    """
    Collect information about a sample.

    Either 'lumi' or 'x_sec' and 'n_events' must be given

    :param name:        str
    :param is_data:     bool (default: False)
    :param is_signal:   bool (default: False)
    :param lumi:        float
    :param x_sec:       float
    :param n_events:    int
    :param legend:      str (used to group samples as well, default: name)
    :param n_events:    int
    :param input_files: list of str

    Optional parameters for cmsRun configs:
    :param output_file:     str (event content out)
    :param cmsRun_builtin:  dict (variable to be attact to builtin of a config)
    :param cmsRun_add_lines: list of str (appended to cmsRun config)
    :param cmsRun_args:     list of str (command line arguments for cmsRun)
    """

    def __init__(self, **kws):
        self.__dict__.update({
            'is_data': False,
            'is_signal': False,
            'x_sec': 0.,
            'n_events': 0,
            'lumi': 0.,
            'legend': '',
            'input_files': [],
            'output_file': '',
            'cmsRun_builtin': {},
            'cmsRun_add_lines': [],
            'cmsRun_args': [],
        })
        self.__dict__.update(kws)
        # check/correct input
        assert(not(self.is_data and self.is_signal))  # both is forbidden!
        if not getattr(self, 'name', 0):
            self.name = self.__class__.__name__
        assert isinstance(self.cmsRun_add_lines, list)
        assert isinstance(self.cmsRun_args, list)
        assert isinstance(self.cmsRun_builtin, dict)
        assert (isinstance(self.input_files, list)
                or isinstance(self.input_files, tuple))
        if self.x_sec and self.n_events:
            self.lumi = self.n_events / float(self.x_sec)
        if not self.lumi:
            monitor.message(
                self.name,
                'WARNING lumi or (x_sec and n_events) seems to be undefined.'
            )
        if not self.legend:
            self.legend = self.name


def _check_n_load(field):
    if inspect.isclass(field) and issubclass(field, Sample):
        smp = field()
        if hasattr(smp, 'enable'):
            if smp.enable:
                return {smp.name: smp}
        elif settings.default_enable_sample:
            return {smp.name: smp}
    return {}


def load_samples(module):
    """
    Get sample instances from a module.

    :param module: modules to import samples from
    :type  module: module
    :returns:      dict of sample classes
    """
    samples = {}
    if isinstance(module, collections.Iterable):
        for mod in module:
            samples.update(load_samples(mod))
    else:
        for name in dir(module):
            if name[0] == '_':
                continue
            field = getattr(module, name)
            try:                    # handle iterable
                for f in field:
                    samples.update(_check_n_load(f))
            except TypeError:       # not an iterable
                samples.update(_check_n_load(field))
    return samples


def generate_samples(in_filenames, in_path='', out_path=''):
    """
    Generates samples for analysis.all_samples.

    The input filename without suffix will be taken as sample name.

    :param in_filenames:    names of inputfiles
    :param in_path:         input path
    :param out_path:        output path
    :returns:               dict of sample classes
    """
    if type(in_filenames) is str:
        in_filenames = [in_filenames]
    samples = {}
    for fname in in_filenames:
        basename    = os.path.basename(fname)
        samplename  = os.path.splitext(basename)[0]
        class sample_subclass(Sample):
            name = samplename
            lumi = 1.
            input_files = in_path + fname
            output_file = out_path
        samples[samplename] = sample_subclass
    return samples


def generate_samples_glob(glob_path, out_path):
    """Globs for files and creates according samples."""
    in_filenames = glob.glob(glob_path)
    in_filenames = itertools.imap(
        lambda t: 'file:' + t,  # prefix with 'file:' for cmssw
        in_filenames
    )
    return generate_samples(
        in_filenames, 
        '',
        out_path
    )


########################################################## SampleNormalizer ###
import analysis
import generators as gen

from varial.toolinterface import Tool
from rendering import default_decorators


class SampleNormalizer(Tool):
    """
    Normalize MC cross sections.

    With this tool all MC cross-section can be normalized to data, using one
    specific distribution. *Before* and *after* plots are stored as plots. The
    resulting factor is stored as result of this tool.

    :param filter_keyfunc:  lambda, keyfunction with one argument
    :param x_range_tuple:
    :param name:            str, tool name
    """
    can_reuse = False

    def __init__(self, filter_keyfunc, x_range_tuple, name=None):
        super(SampleNormalizer, self).__init__(name)
        self.filter_keyfunc = filter_keyfunc
        self.x_range = x_range_tuple

    def get_histos_n_factor(self):
        mcee, data = next(gen.fs_mc_stack_n_data_sum(
            self.filter_keyfunc
        ))
        dh, mh = data.histo, mcee.histo
        bins = tuple(dh.FindBin(x) for x in self.x_range)
        factor = dh.Integral(*bins) / mh.Integral(*bins)
        canv = next(gen.canvas(
            ((mcee, data),),
            default_decorators
        ))
        return factor, canv

    def run(self):
        # before
        factor, canv = self.get_histos_n_factor()
        next(gen.save_canvas_lin_log([canv], lambda _: 'before'))

        # alter samples
        for s in analysis.mc_samples().itervalues():
            s.lumi /= factor
            s.x_sec /= factor
        for a in analysis.fs_aliases:
            a.lumi /= factor

        # after
        _, canv = self.get_histos_n_factor()
        next(gen.save_canvas_lin_log([canv], lambda _: 'after'))

        self.result = wrappers.FloatWrapper(
            factor,
            name='Lumi factor'
        )
