"""
All concrete tools and toolschains that are predefined in varial are here.

The tool-baseclass and toolschains are defined in :ref:`toolinterface-module`.
More concrete tools that are defined in seperate modules are:

=================== ==========================
FSPlotter           :ref:`plotter-module`
RootFilePlotter     :ref:`plotter-module`
Webcreator          :ref:`webcreator-module`
CmsRunProxy         :ref:`cmsrunproxy-module`
FwliteProxy         :ref:`fwliteproxy-module`
=================== ==========================
"""

import itertools
import glob
import os
import shutil
import time
import subprocess

import analysis
import dbio
import diskio
import generators as gen
import settings
import wrappers

from toolinterface import \
    Tool, \
    ToolChain, \
    ToolChainIndie, \
    ToolChainVanilla
from cmsrunproxy import CmsRunProxy
from fwliteproxy import FwliteProxy
from plotter import Plotter, RootFilePlotter
from webcreator import WebCreator


class HistoLoader(Tool):
    """
    Loads histograms from any rootfile or from fileservice.

    :param name:                str, tool name
    :param pattern:             str, pattern for filesearch, e.g. ``*.root``,
                                default: None (load from fileservice)
    :param filter_keyfunc:      lambda, keyfunction with one argument,
                                default: ``None`` (load all histograms)
    :param hook_loaded_histos:  generator to be applied after loading,
                                default: ``None``
    :param io:                  io module,
                                default: ``dbio``
    """
    def __init__(self,
                 pattern=None,
                 filter_keyfunc=None,
                 hook_loaded_histos=None,
                 io=dbio,
                 name=None):
        super(HistoLoader, self).__init__(name)
        self.pattern = pattern
        self.filter_keyfunc = filter_keyfunc
        self.hook_loaded_histos = hook_loaded_histos
        self.io = io

    def run(self):
        if self.pattern:
            wrps = gen.dir_content(self.pattern)
            wrps = itertools.ifilter(self.filter_keyfunc, wrps)
            wrps = gen.sort(wrps)
            wrps = gen.load(wrps)
        else:
            wrps = gen.fs_filter_active_sort_load(self.filter_keyfunc)
        if self.hook_loaded_histos:
            wrps = self.hook_loaded_histos(wrps)
        self.result = list(wrps)


class CopyTool(Tool):
    """
    Copy contents of a directory. Preserves .htaccess files.

    :param dest:            str, destination path
    :param src:             str, source path,
                            default: ``''`` (copy everything in same directory)
    :param ignore:          list,
                            default:
                            ``("*.root", "*.pdf", "*.eps", "*.log", "*.info")``
    :param wipe_dest_dir:   bool, default: ``True``
    :param name:            str, tool name
    """
    def __init__(self, dest, src='',
                 ignore=("*.root", "*.pdf", "*.eps", "*.log", "*.info"),
                 wipe_dest_dir=True,
                 name=None):
        super(CopyTool, self).__init__(name)
        self.dest = dest
        self.src = src
        self.ignore = ignore
        self.wipe_dest_dir = wipe_dest_dir

    def run(self):
        src = os.path.abspath(self.src or os.path.join(self.cwd, '..'))
        dest = os.path.abspath(self.dest)

        # check for htaccess and copy it to src dirs
        htaccess = os.path.join(dest, '.htaccess')
        if os.path.exists(htaccess):
            for path, _, _ in os.walk(src):
                shutil.copy2(htaccess, path)

        # clean dest dir and copy
        if self.wipe_dest_dir:
            for f in glob.glob(dest + '/*'):
                shutil.rmtree(f, True)
        ign_pat = shutil.ignore_patterns(*self.ignore)
        for f in glob.glob(src + '/*'):
            if os.path.isdir(f):
                f = os.path.basename(f)
                shutil.copytree(
                    os.path.join(src, f),
                    os.path.join(dest, f),
                    ignore=ign_pat,
                )
            else:
                shutil.copy2(f, dest)


class ZipTool(Tool):
    """
    Zip-compress a target folder.

    :param abs_path:    str, absolute path of tool to be zipped
    """
    def __init__(self, abs_path):
        super(ZipTool, self).__init__(None)
        self.abs_path = abs_path

    def run(self):
        p = os.path.join(settings.varial_working_dir, self.abs_path)
        os.system(
            'zip -r %s %s' % (p, p)
        )


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
            Plotter.defaults_attrs['canvas_decorators']
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


class GitTagger(Tool):
    can_reuse = False

    def __init__(self, logfilename='GITTAGGER_LOG.txt'):
        super(GitTagger, self).__init__()
        self.logfilename = logfilename

    def run(self):
        if os.system('git diff --quiet') or os.system('git diff --cached --quiet'):
            os.system('git status')
            commit_msg = raw_input('Please give commit message (empty := amend commit, -no := no commit): ')
            if commit_msg == "":
                previous_commit_msg = subprocess.check_output('git log -1 --pretty=%B', shell=True)
                previous_commit_msg = previous_commit_msg.replace('\n', '')
                os.system('git commit -a --amend -m "{0}"'.format(previous_commit_msg))
                with open(self.logfilename) as readf:
                    lines = readf.readlines()
                    latest_tag = lines[-1].split()[0]
                with open(self.logfilename, "w") as writef:
                    writef.writelines([item for item in lines[:-1]])
                    writef.write(time.strftime(latest_tag + " %Y%m%dT%H%M%S ",time.localtime()) + previous_commit_msg + '\n')
                os.system('git tag -af "plot_v{0}" -m "Automatically created tag version {0}"'.format(latest_tag))

            elif commit_msg.startswith('-a'):
                os.system('git commit -a --amend -m "From plot.py: {0}"'.format(commit_msg[3:]))
                with open(self.logfilename) as readf:
                    lines = readf.readlines()
                    latest_tag = lines[-1].split()[0]
                with open(self.logfilename, "w") as writef:
                    writef.writelines([item for item in lines[:-1]])
                    writef.write(time.strftime(latest_tag + " %Y%m%dT%H%M%S ",time.localtime()) + commit_msg[3:] + '\n')

            elif commit_msg == "-no":
                pass

            elif commit_msg:
                with open(self.logfilename, 'a+') as logf:
                    try:
                        lastline = logf.readlines()[-1]
                        latest_tag = lastline.split()[0]
                        version_split = latest_tag.split('.')
                        if len(version_split) != 2:
                            raise NameError('Invalid tag in {0}!'.format(logfilename))
                        new_subtag = int(version_split[-1])+1
                        new_tag = version_split[0]+'.'+str(new_subtag)
                    except IndexError:
                        print '!!Index Error!!'
                        new_tag = '1.0'
                    except ValueError:
                        raise NameError('Invalid version tag in {0}!'.format(logfilename))
                    settings.git_tag = new_tag
                    os.system('git commit -am "From plot.py: {0}"'.format(commit_msg))
                    os.system('git tag -af "plot_v{0}" -m "Automatically created tag version {0}"'.format(new_tag))
                    logf.write(time.strftime(new_tag + " %Y%m%dT%H%M%S ",time.localtime()) + commit_msg + '\n')

    # def check_version(self, filename, index=-1):
    #     filename.seek(0,0)
    #     try:
    #         lastline = filename.readlines[index]
    #     except:



def mk_rootfile_plotter(name="RootFilePlots",
                        pattern='*.root',
                        flat=False,
                        plotter_factory=None,
                        combine_files=False,
                        filter_keyfunc=None):
    """
    Make a plotter chain that plots all content of all rootfiles in cwd.

    :param name:                str, name of the folder in which the output is
                                stored
    :param pattern:             str, search pattern for rootfiles,
                                default: ``*.root``
    :param flat:                bool, flatten the rootfile structure
                                default: ``False``
    :param plotter_factory:     factory function for RootFilePlotter
                                default: ``None``
    :param combine_files:       bool, plot same histograms across rootfiles
                                into the same canvas. Does not work together
                                with ``flat`` option,
                                default: ``False``
    """
    if combine_files:
        plotters = [RootFilePlotter(
            pattern, plotter_factory, flat, name, filter_keyfunc)]
        tc = ToolChain(name, plotters)
    else:
        plotters = list(
            RootFilePlotter(
                f,
                plotter_factory,
                flat,
                f[:-5].split('/')[-1],
                filter_keyfunc,
            )
            for f in glob.iglob(pattern)
        )
        tc = ToolChain(name, [ToolChain(name, plotters)])
    return tc






