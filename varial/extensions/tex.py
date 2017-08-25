"""
Tools connected to the use of (La)Tex.
"""

import shutil
import os
import time
import pprint

from varial.toolinterface import Tool


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
                 images={},
                 plain_files={},
                 include_str='%s',
                 dest_dir=None,
                 dest_dir_name=None,
                 do_hash=True,
                 time_hash=False,
                 name=None):
        super(TexContent, self).__init__(name)
        self.images = images
        self.tex_files = plain_files
        self.include_str = include_str
        self.dest_dir = dest_dir
        self.dest_dir_name = dest_dir_name
        self.do_hash = do_hash
        self.time_hash = time_hash

    def _join(self, *args):
        return os.path.join(self.dest_dir, *args)

    # @staticmethod
    def _hashified_filename(self, path):
        bname, _ = os.path.splitext(os.path.basename(path))
        if self.time_hash:
            hash_str = '_' + time.strftime('%m%d%H')
        elif self.do_hash:
            hash_str = '_' + hex(hash(os.path.dirname(path)))[-7:]
        else:
            hash_str = ''
        return bname + hash_str

    def initialize(self):
        if not self.dest_dir:
            self.dest_dir = self.cwd
        if not self.dest_dir_name:
            p_elems = self.dest_dir.split('/')
            self.dest_dir_name = p_elems[-1] or p_elems[-2]

    def copy_image_files(self):
        for blockname, blockfiles in self.images.iteritems():
            if not all(self.lookup_filename(b, raise_on_empty_path=False) for b in blockfiles):
                missing_files = list(b for b in blockfiles if not self.lookup_filename(b, raise_on_empty_path=False))
                self.message('WARNING the following files in %s do not exist: %s' % (blockname, str(missing_files)))
                continue

            hashified_and_path = list(
                (self._hashified_filename(bf), self.lookup_filename(bf, raise_on_empty_path=False)) for bf in blockfiles
                if self.lookup_filename(bf, raise_on_empty_path=False)
            )

            # make block file
            with open(self._join(blockname+'.tex'), 'w') as f:

                for hashified, path in hashified_and_path:

                    # prepare image
                    p, ext = os.path.splitext(path)
                    if ext == '.eps':
                        os.system('ps2pdf -dEPSCrop %s.eps %s.pdf' % (p, p))
                        ext = '.pdf'
                    elif not ext in ('.pdf', '.png'):
                        raise RuntimeError(
                            'Only .eps, .pdf and .png images are supported.')

                    # copy image file
                    img_dest = blockname + '_' + hashified.replace('.', '-').replace('[', '').replace(']', '')
                    shutil.copy(p+ext, self._join(img_dest+ext))

                    # write tex include
                    inc_dest = os.path.join(self.dest_dir_name, img_dest)
                    f.write(self.include_str % inc_dest + '\n')

    def copy_plain_files(self):
        tex_dict = self.tex_files() if callable(self.tex_files) else self.tex_files
        for fname, path in tex_dict.iteritems():
            if not self.lookup_filename(path, raise_on_empty_path=False):
                self.message('WARNING file %s does not exist' % path)
                continue
            shutil.copy(self.lookup_filename(path, raise_on_empty_path=False), self._join(fname))

    def run(self):
        self.initialize()
        self.copy_image_files()
        self.copy_plain_files()
