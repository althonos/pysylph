import os
import unittest
import gzip
import importlib.resources
from contextlib import nullcontext

from pysylph import Database, Sketcher


def read_fasta_seq(file):
    file.readline() # skip header
    return "".join(map(str.strip, file))


class TestSketcher(unittest.TestCase):

    def test_sketch_genome(self):
        # load reference database
        if hasattr(importlib.resources, "files"):
            handler = nullcontext(importlib.resources.files(__package__).joinpath("clodf13.syldb"))
        else:
            handler = importlib.resources.path(__package__, "clodf13.syldb")
        with handler as path:
            db = Database.load(os.fspath(path))
        # load seq
        if hasattr(importlib.resources, "files"):
            handler = nullcontext(importlib.resources.files(__package__).joinpath("clodf13.fna.gz"))
        else:
            handler = importlib.resources.path(__package__, "clodf13.fna.gz")
        with handler as path:
            with gzip.open(path, "rt") as f:
                seq = read_fasta_seq(f)
        # sketch sequence
        sketcher = Sketcher()
        sketch = sketcher.sketch_genome("CloDF13", [seq])
        # compare to CLI output
        self.assertEqual(sketch.name, "CloDF13")
        self.assertEqual(sketch.k, db[0].k)
        self.assertEqual(sketch.c, db[0].c)
        self.assertEqual(sketch.kmers, db[0].kmers)

    def test_invalid_kmer(self):
        self.assertRaises(ValueError, Sketcher, k=8)