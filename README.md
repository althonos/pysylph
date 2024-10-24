# 🕊️ Pysylph [![Stars](https://img.shields.io/github/stars/althonos/pysylph.svg?style=social&maxAge=3600&label=Star)](https://github.com/althonos/pysylph/stargazers)

*[PyO3](https://pyo3.rs/) bindings and Python interface to [sylph](https://github.com/bluenote-1577/sylph), an ultrafast method for containment ANI querying and taxonomic profiling.*

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/pysylph/test.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/pysylph/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/pysylph/branch/main.svg?style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/pysylph/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/mit/)
[![PyPI](https://img.shields.io/pypi/v/pysylph.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/pysylph)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/pysylph?style=flat-square&maxAge=3600&logo=anaconda)](https://anaconda.org/bioconda/pysylph)
[![AUR](https://img.shields.io/aur/version/python-pysylph?logo=archlinux&style=flat-square&maxAge=3600)](https://aur.archlinux.org/packages/python-pysylph)
[![Wheel](https://img.shields.io/pypi/wheel/pysylph.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/pysylph/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/pysylph.svg?style=flat-square&maxAge=600)](https://pypi.org/project/pysylph/#files)
[![Python Implementations](https://img.shields.io/pypi/implementation/pysylph.svg?style=flat-square&maxAge=600&label=impl)](https://pypi.org/project/pysylph/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/pysylph/)
[![Mirror](https://img.shields.io/badge/mirror-LUMC-001158?style=flat-square&maxAge=2678400)](https://git.lumc.nl/mflarralde/pysylph/)
[![Issues](https://img.shields.io/github/issues/althonos/pysylph.svg?style=flat-square&maxAge=600)](https://github.com/althonos/pysylph/issues)
[![Docs](https://img.shields.io/readthedocs/pysylph/latest?style=flat-square&maxAge=600)](https://pysylph.readthedocs.io)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/pysylph/blob/master/CHANGELOG.md)
[![Downloads](https://img.shields.io/pypi/dm/pysylph?style=flat-square&color=303f9f&maxAge=86400&label=downloads)](https://pepy.tech/project/pysylph)

## 🗺️ Overview

`sylph` is a method developed by [Jim Shaw](https://jim-shaw-bluenote.github.io/)
and [Yun William Yu](https://github.com/yunwilliamyu) for fast and robust
ANI querying or metagenomic profiling for metagenomic shotgun samples. It uses 
a statistical model based on Poisson coverage to compute coverage-adjusted ANI
instead of naive ANI. 

`pysylph` is a Python module, implemented using the [PyO3](https://pyo3.rs/)
framework, that provides bindings to `sylph`. It directly links to the
`sylph` code, which has the following advantages over CLI wrappers:

- **pre-built wheels**: `pysylph` is distributed on PyPI and features
  pre-built wheels for common platforms, including x86-64 and Arm64 UNIX.
- **single dependency**: If your software or your analysis pipeline is
  distributed as a Python package, you can add `pysylph` as a dependency to
  your project, and stop worrying about the `sylph` binary being present on
  the end-user machine.
- **sans I/O**: Everything happens in memory, in Python objects you control,
  making it easier to pass your sequences to `pysylph` without having to write
  them to a temporary file.

*This library is still a work-in-progress, and in an experimental stage, with
API breaks very likely between minor versions*.


## 🔧 Installing

Pysylph can be installed directly from [PyPI](https://pypi.org/project/pysylph/),
which hosts some pre-built CPython wheels for x86-64 platforms, as well as the 
code required to compile from source with Rust and [maturin](https://www.maturin.rs/):
```console
$ pip install pysylph
```

## 🔖 Citation

Pysylph is scientific software, and builds on top of `sylph`. Please cite 
[`sylph`](https://github.com/bluenote-1577/sylph) if you are using it in
an academic work, for instance as:

> `pysylph`, a Python library binding to `sylph` (Shaw & Yu, 2024).


## 💡 Examples

### 🔨 Creating a database

A database is a collection of genomes sketched for fast querying. 

Here is how to create a database into memory, using 
[Biopython](https://github.com/biopython/biopython) to load genomes:

```python
sketcher = pysylph.Sketcher()
sketches = []

for path in pathlib.Path(".").glob("*.fasta"):
    contigs = [ str(record.seq) for record in Bio.SeqIO.parse(path, "fasta") ]
    sketch = sketcher.sketch_genome(name=path.stem, contigs=contigs)
    sketches.append(sketch)

database = pysylph.Database(sketches)
```

`Sketcher` methods are re-entrant and can be used to sketch multiple genomes
in parallel using for instance a [`ThreadPool`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.ThreadPool).

### 📝 Saving a database

The database can be saved to the binary format used by the `sylph` binary as
well:

```python
database.dump("genomes.syldb")
```

### 🗒️ Loading a database

A database previously created with `sylph` can be loaded transparently in 
`pysylph`:

```python
database = pysylph.Database.load("genomes.syldb")
```

### 📊 Sketching a query

Samples must also be sketched before they can be used to query a database.
Here is how to sketch a sample made of single-ended reads stored in FASTQ 
format:

```python
reads = [str(record.seq) for record in Bio.SeqIO.parse("sample.fastq", "fastq")]
sample = sketcher.sketch_single(name="sample", reads=reads)
```

### 🔬 Querying a database

Once a sample has been sketched, it can be used to query a database for ANI
containment or taxonomic profiling:

```python
profiler = pysylph.Profiler()
results = profiler.query(sample, database)   # ANI containment
results = profiler.profile(sample, database) # taxonomic profiling
```

`Profiler` methods are re-entrant and can be used to query a database with
multiple samples in parallel using for instance a 
[`ThreadPool`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.ThreadPool).

## 🔎 See Also

Computing ANI for closed genomes? You may also be interested in
[`pyskani`, a Python package for computing ANI](https://github.com/althonos/pyskani) binding to [`skani`](https://github.com/bluenote-1577/skani), which
was developed by the same authors.

## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug ? Have an enhancement request ? Head over to the
[GitHub issue tracker](https://github.com/althonos/pysylph/issues) if you need
to report or ask something. If you are filing in on a bug, please include as
much information as you can about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

### 🏗️ Contributing

Contributions are more than welcome! See
[`CONTRIBUTING.md`](https://github.com/althonos/pysylph/blob/master/CONTRIBUTING.md)
for more details.


## ⚖️ License

This library is provided under the [MIT License](https://choosealicense.com/licenses/mit/). 
It contains some code included verbatim from the the `sylph` source code, which 
was written by [Jim Shaw](https://jim-shaw-bluenote.github.io/) and is distributed 
under the terms of the [MIT License](https://choosealicense.com/licenses/mit/)
as well. Source distributions of `pysylph` vendors additional sources under their 
own terms using the [`cargo vendor`](https://doc.rust-lang.org/cargo/commands/cargo-vendor.html)
command.

*This project is in no way not affiliated, sponsored, or otherwise endorsed
by the [original `sylph` authors](https://jim-shaw-bluenote.github.io/).
It was developed by [Martin Larralde](https://github.com/althonos/) during his
PhD project at the [Leiden University Medical Center](https://www.lumc.nl/en/)
in the [Zeller team](https://github.com/zellerlab).*
