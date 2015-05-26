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
import diskio
import generators as gen
import pklio
import settings
import wrappers

from toolinterface import \
    Tool, \
    ToolChain, \
    ToolChainIndie, \
    ToolChainVanilla, \
    ToolChainParallel
from plotter import \
    Plotter, \
    RootFilePlotter
from webcreator import \
    WebCreator


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
                 io=pklio,
                 name=None):
        super(HistoLoader, self).__init__(name)
        self.pattern = pattern
        self.filter_keyfunc = filter_keyfunc
        self.hook_loaded_histos = hook_loaded_histos
        self.io = io

    def run(self):
        if self.pattern:
            if not glob.glob(self.pattern):
                self.message('WARNING No input file found for pattern "%s"'
                             % self.pattern)
                wrps = []
            else:
                wrps = gen.dir_content(self.pattern)
                wrps = itertools.ifilter(self.filter_keyfunc, wrps)
                wrps = gen.load(wrps)
                if self.hook_loaded_histos:
                    wrps = self.hook_loaded_histos(wrps)
                wrps = gen.sort(wrps)
        else:
            wrps = gen.fs_filter_active_sort_load(self.filter_keyfunc)
            if self.hook_loaded_histos:
                wrps = self.hook_loaded_histos(wrps)
        self.result = list(wrps)

        if not self.result:
            self.message('WARNING No histograms found.')


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
        self.dest = dest.replace('~', os.getenv('HOME'))
        self.src = src.replace('~', os.getenv('HOME'))
        self.ignore = ignore
        self.wipe_dest_dir = wipe_dest_dir

    def run(self):
        if self.src:
            src = os.path.abspath(self.src)
            src_objs = glob.glob(src)
        elif self.cwd:
            src = os.path.abspath(os.path.join(self.cwd, '..'))
            src_objs = glob.glob(src + '/*')
        else:
            src = os.getcwd()
            src_objs = glob.glob(src + '/*')
        dest = os.path.abspath(self.dest)

        # check for htaccess and copy it to src dirs
        htaccess = os.path.join(dest, '.htaccess')
        if os.path.exists(htaccess):
            for src in src_objs:
                for path, _, _ in os.walk(src):
                    shutil.copy2(htaccess, path)

        # clean dest dir
        if self.wipe_dest_dir:
            src_basenames = list(os.path.basename(p) for p in src_objs)
            for f in glob.glob(dest + '/*'):
                if os.path.isdir(f) and os.path.basename(f) in src_basenames:
                    self.message('INFO Deleting: ' + f)
                    shutil.rmtree(f, True)

        # copy
        ign_pat = shutil.ignore_patterns(*self.ignore)
        for src in src_objs:
            self.message('INFO Copying: ' + src)
            if os.path.isdir(src):
                f = os.path.basename(src)
                shutil.copytree(
                    src,
                    os.path.join(dest, f),
                    ignore=ign_pat,
                )
            else:
                shutil.copy2(src, dest)


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


    def open_file(self, filename):
        try:
            with open(filename) as readf:
                lines = readf.readlines()
                latest_tag = lines[-1].split()[0]
                print latest_tag
                return True
        except IOError:
            return False


    def find_log(self, filename):
        


    def __init__(self, logfilename='GITTAGGER_LOG.txt'):
        super(GitTagger, self).__init__()
        self.logfilename = logfilename
        self.counter = 4
        self.logfound = False


    def run(self):
        if os.system('git diff --quiet') or os.system('git diff --cached --quiet'):
            os.system('git status')
            commit_msg = raw_input('Please give commit message (empty := amend commit, -a <Msg>:= amend commit with new message <Msg>, -no := no commit): ')
            if commit_msg == '':
                previous_commit_msg = subprocess.check_output('git log -1 --pretty=%B', shell=True)
                previous_commit_msg = previous_commit_msg.replace('\n', '')
                with open(self.logfilename) as readf:
                    lines = readf.readlines()
                    latest_tag = lines[-1].split()[0]
                with open(self.logfilename, "w") as writef:
                    writef.writelines([item for item in lines[:-1]])
                    writef.write(time.strftime(latest_tag + " %Y%m%dT%H%M%S ",time.localtime()) + previous_commit_msg + '\n')
                os.system('git commit -a --amend -m "{0}"'.format(previous_commit_msg))
                os.system('git tag -af "plot_v{0}" -m "Automatically created tag version {0}"'.format(latest_tag))

            elif commit_msg.startswith('new_tag '):
                with open(self.logfilename, 'a+') as logf:
                    new_tag = commit_msg.split()[1]
                    if len(new_tag.split('.')) != 2 or not new_tag.split('.')[0].isdigit() or not new_tag.split('.')[1].isdigit():
                        raise NameError('ERROR: invalid tag')
                        return
                    commit_msg = ' '.join(commit_msg.split()[2:])
                    settings.git_tag = new_tag
                    logf.write(time.strftime(new_tag + " %Y%m%dT%H%M%S ",time.localtime()) + commit_msg + '\n')
                    os.system('git commit -am "From GitTagger: {0}"'.format(commit_msg))
                    os.system('git tag -af "plot_v{0}" -m "Automatically created tag version {0}"'.format(new_tag))

            elif commit_msg.startswith('-a'):
                with open(self.logfilename) as readf:
                    lines = readf.readlines()
                    latest_tag = lines[-1].split()[0]
                with open(self.logfilename, "w") as writef:
                    writef.writelines([item for item in lines[:-1]])
                    writef.write(time.strftime(latest_tag + " %Y%m%dT%H%M%S ",time.localtime()) + commit_msg[3:] + '\n')
                os.system('git commit -a --amend -m "From GitTagger: {0}"'.format(commit_msg[3:]))

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
                            return
                        new_subtag = int(version_split[-1])+1
                        new_tag = version_split[0]+'.'+str(new_subtag)
                    except IndexError:
                        print 'Index Error! Set tag to 1.0'
                        new_tag = '1.0'
                    except ValueError:
                        raise NameError('Invalid version tag in {0}!'.format(logfilename))
                        return
                    settings.git_tag = new_tag
                    logf.write(time.strftime(new_tag + " %Y%m%dT%H%M%S ",time.localtime()) + commit_msg + '\n')
                    os.system('git commit -am "From GitTagger: {0}"'.format(commit_msg))
                    os.system('git tag -af "plot_v{0}" -m "Automatically created tag version {0}"'.format(new_tag))

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
                        filter_keyfunc=None,
                        legendnames=None,
                        **kws):
    """
    Make a plotter chain that plots all content of all rootfiles in cwd.

    Additional keywords are forwarded to the plotter instanciation.

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
    def plotter_factory_kws(**kws_fctry):
        kws_fctry.update(kws)
        if plotter_factory:
            return plotter_factory(**kws_fctry)
        else:
            return Plotter(**kws_fctry)

    if kws:
        new_plotter_factory = plotter_factory_kws
    else:
        new_plotter_factory = plotter_factory

    if combine_files:
        plotters = [RootFilePlotter(
            pattern,
            new_plotter_factory,
            flat,
            name,
            filter_keyfunc,
            legendnames
        )]
        tc = ToolChain(name, plotters)
    else:
        plotters = list(
            RootFilePlotter(
                f,
                new_plotter_factory,
                flat,
                f[:-5].split('/')[-1],
                filter_keyfunc,
                legendnames
            )
            for f in glob.iglob(pattern)
        )
        tc = ToolChain(name, [ToolChain(name, plotters)])
    return tc






