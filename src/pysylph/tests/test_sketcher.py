import contextlib
import os
import unittest
import gzip
import importlib.resources
from contextlib import nullcontext

from pysylph import Database, Sketcher, SampleSketch


def every_n(it, n=4):
    try:
        while True:
            yield next(it)
            for i in range(n-1):
                next(it)
    except StopIteration:
        return

def read_fasta_seq(file):
    file.readline()  # skip header
    return "".join(map(str.strip, file))

def read_fastq(file):
    file.readline()  # skip header
    return [ line.strip() for line in every_n(file, n=4)  ]

@contextlib.contextmanager
def get_path(name):
    if hasattr(importlib.resources, "files"):
        handler = nullcontext(
            importlib.resources.files(__package__).joinpath(name)
        )
    else:
        handler = importlib.resources.path(__package__, name)
    with handler as f:
        yield f


class TestSketcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load reference database
        with get_path("clodf13.syldb") as path:
            cls.db = Database.load(path)
        # load seq
        with get_path("clodf13.fna.gz") as path:
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

    def test_sketch_paired(self):
        with get_path("k12_R1.fq.gz") as path:
            with gzip.open(path, "rt") as f:
                r1 = read_fastq(f)
        with get_path("k12_R2.fq.gz") as path:
            with gzip.open(path, "rt") as f:
                r2 = read_fastq(f)

        sketcher = Sketcher()
        sketch = sketcher.sketch_paired("k12", r1, r2)
        
        with get_path("k12.paired.sylsp") as path:
            expected = SampleSketch.load(path)
        self.assertEqual(sketch.kmer_counts, expected.kmer_counts)
        