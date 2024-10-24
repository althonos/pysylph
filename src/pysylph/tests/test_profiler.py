import os
import unittest
import gzip
import importlib.resources
from contextlib import nullcontext

from pysylph import Database, SampleSketch, Profiler


class TestProfiler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load reference database
        if hasattr(importlib.resources, "files"):
            handler = nullcontext(importlib.resources.files(__package__).joinpath("ecoli.syldb"))
        else:
            handler = importlib.resources.path(__package__, "ecoli.syldb")
        with handler as path:
            cls.db = Database.load(path)
        # load seq
        if hasattr(importlib.resources, "files"):
            handler = nullcontext(importlib.resources.files(__package__).joinpath("o157.sylsp"))
        else:
            handler = importlib.resources.path(__package__, "o157.sylsp")
        with handler as path:
            cls.sample = SampleSketch.load(path)


    def test_query(self):
        # sketch sequence
        profiler = Profiler(self.db)
        results = profiler.query(self.sample)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].ani, 99.73)
        self.assertEqual(results[0].ani_naive, 96.02)

        self.assertEqual(results[1].ani, 98.25)
        self.assertEqual(results[1].ani_naive, 94.29)
        
        self.assertEqual(results[2].ani, 98.16)
        self.assertEqual(results[2].ani_naive, 94.26)

    def test_profile(self):
        # sketch sequence
        profiler = Profiler(self.db)
        results = profiler.profile(self.sample)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].ani, 99.73)
        self.assertEqual(results[0].ani_naive, 96.02)
        self.assertEqual(results[0].sequence_abundance, 100.0)
        self.assertEqual(results[0].taxonomic_abundance, 100.0)

    def test_invalid_kmer(self):
        self.assertRaises(ValueError, Sketcher, k=8)