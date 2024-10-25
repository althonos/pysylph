import os
import unittest
import gzip
import importlib.resources
from contextlib import nullcontext

from pysylph import Database, Sketcher


def read_fasta_seq(file):
    file.readline()  # skip header
    return "".join(map(str.strip, file))


class TestSketcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load reference database
        if hasattr(importlib.resources, "files"):
            handler = nullcontext(
                importlib.resources.files(__package__).joinpath("clodf13.syldb")
            )
        else:
            handler = importlib.resources.path(__package__, "clodf13.syldb")
        with handler as path:
            cls.db = Database.load(os.fspath(path))
        # load seq
        if hasattr(importlib.resources, "files"):
            handler = nullcontext(
                importlib.resources.files(__package__).joinpath("clodf13.fna.gz")
            )
        else:
            handler = importlib.resources.path(__package__, "clodf13.fna.gz")
        with handler as path:
            with gzip.open(path, "rt") as f:
                cls.seq = read_fasta_seq(f)

    def test_sketch_genome_str(self):
        sketcher = Sketcher()
        sketch = sketcher.sketch_genome("CloDF13", [self.seq])
        self.assertEqual(sketch.name, "CloDF13")
        self.assertEqual(sketch.k, self.db[0].k)
        self.assertEqual(sketch.c, self.db[0].c)
        self.assertEqual(sketch.kmers, self.db[0].kmers)

    def test_invalid_kmer(self):
        self.assertRaises(ValueError, Sketcher, k=8)

    def test_sketch_genome_bytes(self):
        b = self.seq.encode()
        sketcher = Sketcher()
        sketch = sketcher.sketch_genome("CloDF13", [b])
        self.assertEqual(sketch.name, "CloDF13")
        self.assertEqual(sketch.k, self.db[0].k)
        self.assertEqual(sketch.c, self.db[0].c)
        self.assertEqual(sketch.kmers, self.db[0].kmers)

    def test_sketch_genome_memoryview(self):
        b = self.seq.encode()
        m = memoryview(b)
        sketcher = Sketcher()
        sketch = sketcher.sketch_genome("CloDF13", [m])
        self.assertEqual(sketch.name, "CloDF13")
        self.assertEqual(sketch.k, self.db[0].k)
        self.assertEqual(sketch.c, self.db[0].c)
        self.assertEqual(sketch.kmers, self.db[0].kmers)
