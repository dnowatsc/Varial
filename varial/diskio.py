"""
Read/Write wrappers to disk.

On disk, a wrapper is represented by a .info file. If it contains root objects,
there's a .root file with the same name in the same directory.
"""


import glob
import resource
from os.path import abspath, basename, dirname, exists, join
from ast import literal_eval
from itertools import takewhile
from ROOT import TFile, TDirectory, TH1, TObject, TTree

import history
import monitor
import sample
import settings
import wrappers


# TODO: IOError problem with to many open file descriptors:
# TODO: http://mihalop.blogspot.gr/2014/05/python-subprocess-and-file-descriptors.html
_n_file_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_n_file_limit, _n_file_limit))
except ValueError:
    pass


class NoDictInFileError(RuntimeError): pass
class NoObjectError(RuntimeError): pass
class NoHistogramError(RuntimeError): pass


################################################################# file refs ###
class _BlockMaker(dict):
    def __enter__(self):
        global _in_a_block
        _in_a_block = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _in_a_block
        _in_a_block = False
        for filename, in _block_of_open_files:
            file_handle = _open_root_files.pop(filename)
            file_handle.Close()


_open_root_files = {}
_block_of_open_files = []
_in_a_block = False
block_of_files = _BlockMaker()


def get_open_root_file(filename):
    if filename in _open_root_files:
        file_handle = _open_root_files[filename]
    else:
        if len(_open_root_files) > settings.max_open_root_files:
            monitor.message(
                'diskio',
                'WARNING to many open root files. Closing all. '
                'Please check for lost histograms. '
                '(Use hist.SetDirectory(0) to keep them)'
            )
            close_open_root_files()
        file_handle = TFile.Open(filename, 'READ')
        if (not file_handle) or file_handle.IsZombie():
            raise RuntimeError('Cannot open file with root: "%s"' % filename)
        _open_root_files[filename] = file_handle
        if _in_a_block:
            _block_of_open_files.append(filename)
    return file_handle


def close_open_root_files():
    for name, file_handle in _open_root_files.iteritems():
        file_handle.Close()
    _open_root_files.clear()


def close_root_file(filename):
    if not filename[-5:] == '.root':
        filename += '.root'
    if filename in _open_root_files:
        _open_root_files[filename].Close()
        del _open_root_files[filename]


##################################################### read / write wrappers ###
use_analysis_cwd = True
_save_log = {}


def prepare_filename(wrp, filename):
    if not filename:
        filename = wrp.name
    if use_analysis_cwd:
        filename = join(analysis.cwd, filename)
    if filename[-5:] == '.info':
        filename = filename[:-5]
    return filename


def write(wrp, filename=None, suffices=(), mode='RECREATE'):
    """Writes wrapper to disk, including root objects."""
    if settings.git_tag: 
        wrp.git_tag = settings.git_tag
    filename = prepare_filename(wrp, filename)
    # check for overwriting something
    if filename in _save_log:
        monitor.message(
            'diskio.write',
            'WARNING Overwriting file from this session: %s' % filename
        )
    else:
        _save_log[filename] = True
    # save with suffices
    for suffix in suffices:
        wrp.primary_object().SaveAs(filename + suffix)
    # WrapperWrapper: store others first
    if isinstance(wrp, wrappers.WrapperWrapper):
        _write_wrapperwrapper(wrp, filename)
    # write root objects (if any)
    if any(isinstance(o, TObject) for o in wrp.__dict__.itervalues()):
        wrp.root_filename = basename(filename+'.root')
        f = TFile.Open(filename+'.root', mode)
        f.cd()
        _write_wrapper_objs(wrp, f)
        f.Close()
    # write wrapper infos
    with open(filename+'.info', 'w') as f:
        _write_wrapper_info(wrp, f)
    _clean_wrapper(wrp)


def _write_wrapper_info(wrp, file_handle):
    #"""Serializes Wrapper to python code dict."""
    history, wrp.history = wrp.history, str(wrp.history)
    file_handle.write(wrp.pretty_writeable_lines() + ' \n\n')
    file_handle.write(wrp.history + '\n')
    wrp.history = history


def _write_wrapper_objs(wrp, file_handle):
    #"""Writes root objects on wrapper to disk."""
    wrp.root_file_obj_names = {}
    if isinstance(wrp, wrappers.FileServiceWrapper):
        dirfile = file_handle.mkdir(wrp.name, wrp.name)
        dirfile.cd()
        for key, value in wrp.__dict__.iteritems():
            if not isinstance(value, TObject):
                continue
            value.Write()
            wrp.root_file_obj_names[key] = value.GetName()
        dirfile.Close()
    else:
        for key, value in wrp.__dict__.iteritems():
            if not isinstance(value, TObject):
                continue
            dirfile = file_handle.mkdir(key, key)
            dirfile.cd()
            value.Write()
            dirfile.Close()
            wrp.root_file_obj_names[key] = value.GetName()


def _write_wrapperwrapper(wrp, filename=None):
    if not filename:
        filename = wrp.name
    wrp_names = []
    for w in wrp.wrps:
        name = filename + '_WRPWRP_' + w.name
        wrp_names.append(basename(name))
        write(w, name)
    wrp.wrps = wrp_names


def read(filename):
    """Reads wrapper from disk, including root objects."""
    if filename[-5:] != '.info':
        filename += '.info'
    if use_analysis_cwd:
        filename = join(analysis.cwd, filename)
    with open(filename, 'r') as f:
        info = _read_wrapper_info(f)
    if 'root_filename' in info:
        _read_wrapper_objs(info, dirname(filename))
    klass = getattr(wrappers, info.get('klass'))
    if klass == wrappers.WrapperWrapper:
        p = dirname(filename)
        info['wrps'] = _read_wrapperwrapper(join(p, f) for f in info['wrps'])
    wrp = klass(**info)
    _clean_wrapper(wrp)
    return wrp


def _read_wrapper_info(file_handle):
    #"""Instaciates Wrapper from info file, without root objects."""
    lines = takewhile(lambda l: l!='\n', file_handle)
    lines = (l.strip() for l in lines)
    lines = ''.join(lines)
    info = literal_eval(lines)
    if not type(info) == dict:
        raise NoDictInFileError('Could not read file: '+file_handle.name)
    return info


def _read_wrapper_objs(info, path):
    #"""Reads root objects from disk."""
    root_file = join(path, info['root_filename'])
    obj_paths = info['root_file_obj_names']
    is_fs_wrp = info['klass'] == 'FileServiceWrapper'
    for key, value in obj_paths.iteritems():
        if is_fs_wrp:
            obj = _get_obj_from_file(root_file, info['name'] + '/' + value)
        else:
            obj = _get_obj_from_file(root_file, key + '/' + value)
        if hasattr(obj, 'SetDirectory'):
            obj.SetDirectory(0)
        info[key] = obj


def _read_wrapperwrapper(wrp_list):
    wrps = []
    for fname in wrp_list:
        wrps.append(read(fname))
    return wrps


def _clean_wrapper(wrp):
    del_attrs = ['root_filename', 'root_file_obj_names', 'wrapped_object_key']
    for attr in del_attrs:
        if hasattr(wrp, attr):
            delattr(wrp, attr)


def get(filename, default=None):
    """Reads wrapper from disk if availible, else returns default."""
    fname = join(analysis.cwd, filename) if use_analysis_cwd else filename
    if fname[-5:] == '.info':
        fname = fname[:-5]
    if exists('%s.info' % fname):
        try:
            return read(filename)
        except RuntimeError:
            return default
    else:
        return default


########################################################## i/o with aliases ###
def generate_fs_aliases(file_path, sample_inst):
    """Produces list of all fileservice histograms for registered samples."""
    if not isinstance(sample_inst, sample.Sample):
        raise RuntimeError(
            '2nd arg of generate_fs_aliases must be instance of sample.Sample')
    root_file = get_open_root_file(file_path)
    for ifp, typ in _recursive_path_and_type(root_file, ''):
        yield wrappers.FileServiceAlias(file_path, ifp, typ, sample_inst)


def generate_aliases(glob_path='./*.root'):
    """Looks for root files and produces aliases."""
    for file_path in glob.iglob(glob_path):
        root_file = get_open_root_file(file_path)
        for ifp, typ in _recursive_path_and_type(root_file, ''):
            yield wrappers.Alias(file_path, ifp, typ)


def _recursive_path_and_type(root_dir, in_file_path):
    for key in root_dir.GetListOfKeys():
        if in_file_path:
            key_path = in_file_path + '/' + key.GetName()
        else:
            key_path = key.GetName()
        if key.IsFolder():
            obj = key.ReadObj()
            if isinstance(obj, TTree):
                continue
            for info in _recursive_path_and_type(
                obj,
                key_path
            ):
                yield info
        else:
            yield key_path, key.GetClassName()


def load_bare_object(alias):
    return _get_obj_from_file(
        alias.file_path,
        alias.in_file_path
    )


def load_histogram(alias):
    """Returns a wrapper with a fileservice histogram."""
    histo = load_bare_object(alias)
    if not isinstance(histo, TH1):
        raise NoHistogramError(
            'Loaded object is not of type TH1: %s' % str(histo)
        )
    if not histo.GetSumw2().GetSize():
        histo.Sumw2()
    wrp = wrappers.HistoWrapper(histo, **alias.all_info())
    if isinstance(alias, wrappers.FileServiceAlias):
        histo.SetTitle(alias.legend)
        wrp.history = history.History(
            'FileService(%s, %s)' % (
                alias.in_file_path, alias.sample))
    else:
        info = alias.all_writeable_info()
        del info['klass']
        wrp.history = history.History(
            'RootFile(%s)' % info
        )
    return wrp


def _get_obj_from_file(filename, in_file_path):
    obj = get_open_root_file(filename)
    # browse through file
    for name in in_file_path.split('/'):
        obj_key = obj.GetKey(name)
        if not obj_key:
            raise NoObjectError(
                'I cannot find "%s" in root file "%s"!' % (in_file_path,
                                                           filename))
        obj = obj_key.ReadObj()
    return obj


################################################### write and close on exit ###
import atexit
import analysis


def write_fileservice():
    if not analysis.fs_wrappers:
        return

    fs_wrappers = analysis.fs_wrappers.values()
    write(fs_wrappers[0], filename=settings.fileservice_filename)
    for wrp in fs_wrappers[1:]:
        write(wrp, filename=settings.fileservice_filename, mode='UPDATE')

atexit.register(write_fileservice)
atexit.register(close_open_root_files)
