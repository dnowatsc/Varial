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

from ast import literal_eval
import subprocess
import itertools
import random
import shutil
import glob
import os
import shutil
import time

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


class Runner(ToolChain):
    """Runs tools upon instanciation (including proper folder creation)."""
    def __init__(self, tool, default_reuse=False):
        super(Runner, self).__init__(None, [tool], default_reuse)
        analysis.reset()
        self.run()


class PrintToolTree(Tool):
    """Calls analysis.print_tool_tree()"""
    can_reuse = False

    def run(self):
        analysis.print_tool_tree()


class UserInteraction(Tool):
    def __init__(self,
                 prompt='Hit enter to continue. Kill me otherwise.',
                 eval_result=False,
                 can_reuse=True,
                 name=None):
        super(UserInteraction, self).__init__(name)
        self.prompt = prompt
        self.eval_result = eval_result
        self.can_reuse = can_reuse

    def run(self):
        if self.eval_result:
            self.message('INFO Input will be evaluated as python code.')
        if self.can_reuse:
            self.message('INFO Input might be reused.')
        res = raw_input(self.prompt+' ')
        if self.eval_result:
            res = literal_eval(res)
        self.result = wrappers.Wrapper(input=res)


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
                 raise_on_empty_result=True,
                 io=pklio,
                 name=None):
        super(HistoLoader, self).__init__(name)
        self.pattern = pattern
        self.filter_keyfunc = filter_keyfunc
        self.hook_loaded_histos = hook_loaded_histos
        self.raise_on_empty_result = raise_on_empty_result
        self.io = io

    def run(self):
        if self.pattern:
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
            if self.raise_on_empty_result:
                raise RuntimeError('ERROR No histograms found.')
            else:
                self.message('ERROR No histograms found.')


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
                 name=None,
                 use_rsync=False):
        super(CopyTool, self).__init__(name)
        self.dest = dest.replace('~', os.getenv('HOME'))
        self.src = src.replace('~', os.getenv('HOME'))
        self.ignore = ignore
        self.wipe_dest_dir = wipe_dest_dir
        self.use_rsync = use_rsync

    def def_copy(self, src_objs, dest, ignore):
        ign_pat = shutil.ignore_patterns(*ignore)
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

    def run(self):
        if self.use_rsync:
            self.wipe_dest_dir = False
            self.ignore = list('--exclude='+w for w in self.ignore)
            cp_func = lambda w, x, y: os.system('rsync -avz --delete {0} {1} {2}'.format(
                ' '.join(w), x, ' '.join(y)))
        else:
            cp_func = lambda w, x, y: self.def_copy(w, x, y)

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
        cp_func(src_objs, dest, self.ignore)


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


class CompileTool(Tool):
    """
    Calls make in the directories given in paths.

    If compilation was needed (i.e. the output of make was different from
    "make: Nothing to be done for `all'") wanna_reuse will return False and
    by that cause all following modules to run.

    :param paths:   list of str: paths where make should be invoked
    """
    nothing_done = 'make: Nothing to be done for `all\'.\n'

    def __init__(self, paths):
        super(CompileTool, self).__init__()
        self.paths = paths

    def wanna_reuse(self, all_reused_before_me):
        nothing_compiled_yet = True
        for path in self.paths:
            self.message('INFO Compiling in: ' + path)
            # here comes a workaround: we need to examine the output of make,
            # but want to stream it directly to the console as well. Hence use
            # tee and look at the output after make finished.
            tmp_out = '/tmp/varial_compile_%06i' % random.randint(0, 999999)
            res = subprocess.call(
                # PIPESTATUS is needed to get the returncode from make
                ['make -j 9 | tee %s; test ${PIPESTATUS[0]} -eq 0' % tmp_out],
                cwd=path,
                shell=True,
            )
            if res:
                os.remove(tmp_out)
                raise RuntimeError('Compilation failed in: ' + path)
            if nothing_compiled_yet:
                with open(tmp_out) as f:
                    if not f.readline() == self.nothing_done:
                        nothing_compiled_yet = False
            os.remove(tmp_out)

        return nothing_compiled_yet and all_reused_before_me

    def run(self):
        pass


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

# TODO: in the GITLOGGER_LOG, each tool should get the commit hash from when it was last updated
# i.e. also if it was amended

class GitTagger(Tool):
    can_reuse = False

    def __init__(self, logfilename="GITTAGGER_LOG.txt"):
        super(GitTagger, self).__init__()
        self.logfilename = logfilename
        # self.counter = 4
        # self.logfound = False

    def print_tool_tree(self):
        toollist = []
        toollist.append('+' + analysis.results_base.name)
        for rname in sorted(analysis.results_base.children):
            self._print_tool_tree(toollist, analysis.results_base.children[rname], 0)
        return toollist


    def _print_tool_tree(self, toollist, res, indent):
        toollist.append('    ' + '|   '*indent + '+' + res.name)
        for rname in sorted(res.children):
            self._print_tool_tree(toollist, res.children[rname], indent + 1)

    def new_commit(self, message=''):
        commit_msg = raw_input(message)
        if commit_msg == '':
            return
        elif commit_msg == 'app':
            previous_commit_msg = subprocess.check_output('git log -1 --pretty=%B', shell=True)
            os.system('git commit --amend -am "{0}"'.format(previous_commit_msg)) # 
        else:
            os.system('git commit -am "From GitTagger: {0}"'.format(commit_msg))


    def run(self):
        toollist = self.print_tool_tree()
        files_changed = False
        if os.system('git diff --quiet') or os.system('git diff --cached --quiet'):
            files_changed = True
        if os.path.isfile(analysis.cwd+self.logfilename):
            new_tool = False
            with open(analysis.cwd+self.logfilename, 'r') as logfile:
                lines = logfile.readlines()
                for index, iTool in enumerate(toollist):
                    if iTool != lines[index][:-44]:
                        new_tool = True
                        break
            if new_tool:
                with open(analysis.cwd+self.logfilename, 'w') as logfile:
                    logfile.writelines(toollist)
                self.new_commit("New tool found, if you want to make new commit type a commit message; "\
                      "If you want to amend the latest commit, type 'app'; "\
                      "If you don't want to commit, just press enter: ")

            else:
                commit_msg = raw_input("No new Tool found, want to amend commit? "\
                    "(For no new commit message just press Enter otherwise type new commit message) ")
                if any((commit_msg == i) for i in ['n', 'N', 'no', 'No', 'NO']):
                    print "No commit"
                    return
                elif commit_msg == '':
                    previous_commit_msg = subprocess.check_output('git log -1 --pretty=%B', shell=True)
                    os.system('git commit --amend -am "{0}"'.format(previous_commit_msg))
                else:
                    os.system('git commit -a --amend -m "From GitTagger: {0}"'.format(commit_msg))
                # print "Logfile content: ", lines
        else:
            self.new_commit("No logfile found, if you want to make new commit type a commit message; "\
                      "If you want to amend the latest commit, type 'app'; "\
                      "If you don't want to commit, just press enter: ")
            with open(analysis.cwd+self.logfilename, 'w') as logfile:
                logfile.writelines((i + ' : ' + \
                    subprocess.check_output('git rev-parse --verify HEAD', shell=True))\
                    for i in toollist)


            # os.system(' git status')
            # commit_msg = raw_input('Please give commit message (empty := amend commit, -a <Msg>:= amend commit with new message <Msg>, -no := no commit): ')
            # if commit_msg == '':
            #     previous_commit_msg = subprocess.check_output('git log -1 --pretty=%B', shell=True)
            #     previous_commit_msg = previous_commit_msg.replace('\n', '')
            #     with open(self.logfilename) as readf:
            #         lines = readf.readlines()
            #         latest_tag = lines[-1].split()[0]
            #     with open(self.logfilename, "w") as writef:
            #         writef.writelines([item for item in lines[:-1]])
            #         writef.write(time.strftime(latest_tag + " %Y%m%dT%H%M%S ",time.localtime()) + previous_commit_msg + '\n')
            #     os.system('git commit -a --amend -m "{0}"'.format(previous_commit_msg))
            #     os.system('git tag -af "plot_v{0}" -m "Automatically created tag version {0}"'.format(latest_tag))

            # elif commit_msg.startswith('new_tag '):
            #     with open(self.logfilename, 'a+') as logf:
            #         new_tag = commit_msg.split()[1]
            #         if len(new_tag.split('.')) != 2 or not new_tag.split('.')[0].isdigit() or not new_tag.split('.')[1].isdigit():
            #             raise NameError('ERROR: invalid tag')
            #             return
            #         commit_msg = ' '.join(commit_msg.split()[2:])
            #         settings.git_tag = new_tag
            #         logf.write(time.strftime(new_tag + " %Y%m%dT%H%M%S ",time.localtime()) + commit_msg + '\n')
            #         os.system('git commit -am "From GitTagger: {0}"'.format(commit_msg))
            #         os.system('git tag -af "plot_v{0}" -m "Automatically created tag version {0}"'.format(new_tag))

            # elif commit_msg.startswith('-a'):
            #     with open(self.logfilename) as readf:
            #         lines = readf.readlines()
            #         latest_tag = lines[-1].split()[0]
            #     with open(self.logfilename, "w") as writef:
            #         writef.writelines([item for item in lines[:-1]])
            #         writef.write(time.strftime(latest_tag + " %Y%m%dT%H%M%S ",time.localtime()) + commit_msg[3:] + '\n')
            #     os.system('git commit -a --amend -m "From GitTagger: {0}"'.format(commit_msg[3:]))

            # elif commit_msg == "-no":
            #     pass

            # elif commit_msg:
            #     with open(self.logfilename, 'a+') as logf:
            #         try:
            #             lastline = logf.readlines()[-1]
            #             latest_tag = lastline.split()[0]
            #             version_split = latest_tag.split('.')
            #             if len(version_split) != 2:
            #                 raise NameError('Invalid tag in {0}!'.format(logfilename))
            #                 return
            #             new_subtag = int(version_split[-1])+1
            #             new_tag = version_split[0]+'.'+str(new_subtag)
            #         except IndexError:
            #             print 'Index Error! Set tag to 1.0'
            #             new_tag = '1.0'
            #         except ValueError:
            #             raise NameError('Invalid version tag in {0}!'.format(logfilename))
            #             return
            #         settings.git_tag = new_tag
            #         logf.write(time.strftime(new_tag + " %Y%m%dT%H%M%S ",time.localtime()) + commit_msg + '\n')
            #         os.system('git commit -am "From GitTagger: {0}"'.format(commit_msg))
            #         os.system('git tag -af "plot_v{0}" -m "Automatically created tag version {0}"'.format(new_tag))

    # def check_version(self, filename, index=-1):
    #     filename.seek(0,0)
    #     try:
    #         lastline = filename.readlines[index]
    #     except:

class TexContent(Tool):
    """
    Copies (and converts) content for usage in a tex document.

    For blocks of images, includestatements are printed into .tex files.
    These can be include in the main tex document.

    Image files in eps format are converted to pdf.

    IMPORTANT: absolute paths must be used in ``images`` and ``plain_files``!

    :param images:      ``{'blockname.tex': ['path/to/file1.eps', ...]}``
    :param plain_files: ``{'target_filename.tex': 'path/to/file1.tex', ...}``
    :param include_str: e.g. ``r'\includegraphics[width=0.49\textwidth]
                        {TexContent/%s}'`` where %s will be formatted with the
                        basename of the image
    :param dest_dir:    destination directory (default: tool path)
    """
    def __init__(self,
                 images,
                 plain_files,
                 include_str,
                 dest_dir=None,
                 name=None):
        super(TexContent, self).__init__(name)
        self.images = images
        self.tex_files = plain_files
        self.include_str = include_str
        self.dest_dir = dest_dir

    def _join(self, basename):
        return os.path.join(self.dest_dir, basename)

    @staticmethod
    def _hashified_filename(path):
        bname, ext = os.path.splitext(os.path.basename(path))
        hash_str = '_' + hex(hash(path))[-7:]
        return bname + hash_str

    def copy_image_files(self):
        for blockname, blockfiles in self.images.iteritems():
            hashified_and_path = list(
                (self._hashified_filename(bf), bf) for bf in blockfiles
            )

            # copy image files
            for hashified, path in hashified_and_path:
                p, ext = os.path.splitext(path)
                if ext == '.eps':
                    os.system('ps2pdf -dEPSCrop %s.eps %s.pdf' % (p, p))
                    ext = '.pdf'
                elif not ext in ('.pdf', '.png'):
                    raise RuntimeError(
                        'Only .eps, .pdf and .png images are supported.')
                shutil.copy(p+ext, self._join(hashified+ext))

            # make block file
            with open(self._join(blockname), 'w') as f:
                for hashified, _ in hashified_and_path:
                    f.write(self.include_str % hashified + '\n')

    def copy_plain_files(self):
        for fname, path, in self.tex_files.iteritems():
            shutil.copy(path, self._join(fname))

    def run(self):
        if not self.dest_dir:
            self.dest_dir = self.cwd
        self.copy_image_files()
        self.copy_plain_files()


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
    For running the plotter(s), use a Runner.

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
        tc = RootFilePlotter(
            pattern,
            new_plotter_factory,
            flat,
            name,
            filter_keyfunc,
            legendnames
        )
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
        tc = ToolChainParallel(name, plotters)
    return tc






