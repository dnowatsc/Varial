#!/usr/bin/env python

import os
from ROOT import TH1F
from test_histotoolsbase import TestHistoToolsBase
from varial.wrappers import FileServiceAlias, HistoWrapper, WrapperWrapper
from varial import diskio
from varial import sparseio
from varial import analysis
from varial import settings

class TestSparseio(TestHistoToolsBase):

    def setUp(self):
        super(TestSparseio, self).setUp()
        if not os.path.exists('test_data'):
            os.mkdir('test_data')
        aliases = diskio.generate_aliases(settings.DIR_FILESERVICE+'tt.root')
        self.test_wrps = list(diskio.load_histogram(a) for a in aliases)
        self.name_func = lambda w: w.in_file_path.replace('/', '_')


    def test_bulk_write(self):
        sparseio.bulk_write(
            self.test_wrps, 'test_data', self.name_func, ('.png', '.pdf'))

        # files should exist
        self.assertTrue(os.path.exists('test_data/' + sparseio._infofile))
        self.assertTrue(os.path.exists('test_data/' + sparseio._rootfile))
        for w in self.test_wrps:
            tok = self.name_func(w)
            self.assertTrue(os.path.exists('test_data/%s.png' % tok))
            self.assertTrue(os.path.exists('test_data/%s.pdf' % tok))

    def test_bulk_read_info_dict(self):
        sparseio.bulk_write(
            self.test_wrps, 'test_data', self.name_func, ('.png', '.pdf'))
        read_in = sparseio.bulk_read_info_dict('test_data')

        # verify filenames
        for name, wrp in read_in.iteritems():
            self.assertEqual(name, self.name_func(wrp))

        # assert input info == output info
        dict_out = dict((self.name_func(w), w.pretty_writeable_lines())
                        for w in self.test_wrps)
        dict_inp = dict((self.name_func(w), w.pretty_writeable_lines())
                        for w in read_in.itervalues())
        self.assertDictEqual(dict_out, dict_inp)


import unittest
suite = unittest.TestLoader().loadTestsFromTestCase(TestSparseio)
if __name__ == '__main__':
    unittest.main()