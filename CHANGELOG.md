# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
[Unreleased]: https://github.com/althonos/pysylph/compare/v0.1.2...HEAD


## [v0.1.2] - 2024-11-05
[v0.1.2]: https://github.com/althonos/pysylph/compare/v0.1.1...v0.1.2

### Added
- `Sketcher.sketch_paired` to sketch paired reads.
- `SampleSketch.kmer_counts` to access the k-mer counts of a sketch as a Python `dict`. 
- `deduplicated` and `fpr` parameters to `Sketcher` to control read deduplication.
- Package recipe on the Arch User Repository.


## [v0.1.1] - 2024-10-25
[v0.1.1]: https://github.com/althonos/pysylph/compare/v0.1.0...v0.1.1

### Added
- Type annotations to `pysylph` module.
- Sphinx documentation with API reference hosted on ReadTheDocs.

### Fixed
- Add missing `ProfileResult` class to the `pysylph.lib` module.

### Changed
- Implement reading sequence data using the buffer-protocol in `Sketcher` methods.


## [v0.1.0] - 2024-10-25
[v0.1.0]: https://github.com/althonos/pysylph/compare/3c6d23...v0.1.0

Initial release.
