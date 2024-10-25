extern crate bincode;
extern crate pyo3;
extern crate statrs;
extern crate sylph;

use std::io::Read;
use std::sync::Arc;

use bincode::de::read::IoReader;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyList;
use pyo3::types::PyString;
use pyo3::types::PyType;
use rayon::prelude::*;
use sylph::types::SequencesSketch;

mod exports;
mod pyfile;

// --- Base --------------------------------------------------------------------

#[pyclass(module = "pysylph.lib", frozen, subclass)]
pub struct Sketch;

#[pymethods]
impl Sketch {
    /// `int`: The subsampling rate this sketch was built with.
    #[getter]
    pub fn c(&self) -> PyResult<usize> {
        Err(PyNotImplementedError::new_err("not implemented"))
    }

    /// `int`: The k-mer size this sketch was built with.
    #[getter]
    pub fn k(&self) -> PyResult<usize> {
        Err(PyNotImplementedError::new_err("not implemented"))
    }
}

// --- GenomeSketch ------------------------------------------------------------

/// A (reference) genome sketch.
#[pyclass(module = "pysylph.lib", frozen, extends=Sketch)]
pub struct GenomeSketch {
    sketch: Arc<sylph::types::GenomeSketch>,
}

impl From<Arc<sylph::types::GenomeSketch>> for GenomeSketch {
    fn from(sketch: Arc<sylph::types::GenomeSketch>) -> Self {
        Self { sketch }
    }
}

impl From<sylph::types::GenomeSketch> for GenomeSketch {
    fn from(sketch: sylph::types::GenomeSketch) -> Self {
        Self::from(Arc::new(sketch))
    }
}

#[pymethods]
impl GenomeSketch {
    pub fn __repr__<'py>(slf: PyRef<'py, Self>) -> PyResult<String> {
        Ok(format!("<GenomeSketch name={:?}>", slf.sketch.file_name))
    }

    /// `str`: The name of the genome this sketch originates from.
    #[getter]
    pub fn name(&self) -> &str {
        &self.sketch.file_name
    }

    /// `str`: The description of the genome this sketch originates from.
    #[getter]
    pub fn description(&self) -> &str {
        &self.sketch.first_contig_name
    }

    /// `int`: The total number of nucleotides in the genome.
    #[getter]
    pub fn genome_size(&self) -> usize {
        self.sketch.gn_size
    }

    /// `int`: The subsampling rate this sketch was built with.
    #[getter]
    pub fn c(&self) -> usize {
        self.sketch.c
    }

    /// `int`: The k-mer size this sketch was built with.
    #[getter]
    pub fn k(&self) -> usize {
        self.sketch.k
    }

    /// `int`: The minimum spacing between k-mers used when sketching.
    #[getter]
    pub fn min_spacing(&self) -> usize {
        self.sketch.min_spacing
    }

    /// `list` of `int`: The list of k-mers extracted from the genome.
    #[getter]
    pub fn kmers<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, &self.sketch.genome_kmers)
    }
}

// --- Database ----------------------------------------------------------------

/// A sketch database.
///
/// Sylph databases are simply a sequence of `GenomeSketch` concatenated
/// together to be used for fast profiling or querying.
///
#[pyclass(module = "pysylph.lib", frozen)]
#[derive(Debug, Default)]
pub struct Database {
    sketches: Vec<Arc<sylph::types::GenomeSketch>>,
}

impl From<Vec<Arc<sylph::types::GenomeSketch>>> for Database {
    fn from(sketches: Vec<Arc<sylph::types::GenomeSketch>>) -> Self {
        Self { sketches }
    }
}

impl FromIterator<sylph::types::GenomeSketch> for Database {
    fn from_iter<T: IntoIterator<Item = sylph::types::GenomeSketch>>(iter: T) -> Self {
        let it = iter.into_iter();
        let sketches = it.map(Arc::new).collect::<Vec<_>>();
        Self::from(sketches)
    }
}

#[pymethods]
impl Database {
    #[new]
    #[pyo3(signature = (items = None))]
    pub fn __new__<'py>(items: Option<Bound<'py, PyAny>>) -> PyResult<Self> {
        let mut db = Self::default();
        if let Some(sketches) = items {
            for object in sketches.iter()? {
                let sketch: PyRef<'py, GenomeSketch> = object?.extract()?;
                db.sketches.push(sketch.sketch.clone());
            }
        }
        Ok(db)
    }

    pub fn __len__<'py>(slf: PyRef<'py, Self>) -> usize {
        slf.sketches.len()
    }

    pub fn __getitem__<'py>(slf: PyRef<'py, Self>, item: isize) -> PyResult<Py<GenomeSketch>> {
        let py = slf.py();
        let mut item_ = item;
        if item_ < 0 {
            item_ += slf.sketches.len() as isize;
        }

        if item_ < 0 || item_ >= slf.sketches.len() as isize {
            Err(PyIndexError::new_err(item))
        } else {
            Py::new(
                py,
                PyClassInitializer::from(Sketch)
                    .add_subclass(GenomeSketch::from(slf.sketches[item_ as usize].clone())),
            )
        }
    }

    /// Load a database from a file.
    ///
    /// Arguments:
    ///     file (`str`, `os.PathLike` or file-like handle): The path to the
    ///         file containing the database, or a file-like object open in
    ///         binary mode.
    ///
    /// Example:
    ///     >>> db = pysylph.Database.load("ecoli.syldb")
    ///     >>> db[0].name
    ///     'test_files/e.coli-K12.fasta.gz'
    ///
    #[classmethod]
    #[pyo3(signature = (file))]
    fn load<'py>(cls: &Bound<'py, PyType>, file: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = cls.py();
        // attempt to extract a path, or fall back to reader
        let result: Result<Vec<Arc<sylph::types::GenomeSketch>>, _> = if let Ok(s) = py
            .import_bound(pyo3::intern!(py, "os"))?
            .call_method1(pyo3::intern!(py, "fsdecode"), (file,))
        {
            let path = s.downcast::<PyString>()?;
            let f = std::fs::File::open(path.to_str()?)?;
            let reader = std::io::BufReader::new(f);
            bincode::deserialize_from(reader)
        } else {
            let f = self::pyfile::PyFileRead::from_ref(&file)?;
            let reader = std::io::BufReader::new(f);
            bincode::deserialize_from(reader)
        };
        // handle loader error
        match result {
            Ok(sketches) => Ok(Database::from(sketches)),
            Err(e) => match *e {
                bincode::ErrorKind::Io(io) => Err(io.into()),
                bincode::ErrorKind::InvalidUtf8Encoding(e) => Err(e.into()),
                other => Err(PyValueError::new_err(format!(
                    "failed to load db: {:?}",
                    other
                ))),
            },
        }
    }

    /// Dump a database to a path.
    fn dump<'py>(slf: PyRef<'py, Self>, path: PyBackedStr) -> PyResult<()> {
        let f = std::fs::File::create(&*path)?;

        bincode::serialize_into(f, &slf.sketches).unwrap();
        Ok(())
    }
}

// --- Database Reader ---------------------------------------------------------

struct DatabaseReader<R: Read> {
    reader: bincode::de::Deserializer<
        IoReader<R>,
        bincode::config::WithOtherTrailing<
            bincode::config::WithOtherIntEncoding<
                bincode::config::DefaultOptions,
                bincode::config::FixintEncoding,
            >,
            bincode::config::AllowTrailing,
        >,
    >,
    length: usize,
}

impl<R: Read> DatabaseReader<R> {
    fn new(r: R) -> Result<Self, bincode::Error> {
        use bincode::Options;
        let mut reader = bincode::de::Deserializer::with_reader(
            r,
            bincode::config::DefaultOptions::new()
                .with_fixint_encoding()
                .allow_trailing_bytes(),
        );
        let length: usize = serde::Deserialize::deserialize(&mut reader).unwrap_or(0);
        Ok(Self { reader, length })
    }
}

impl<R: Read> std::iter::Iterator for DatabaseReader<R> {
    type Item = Result<sylph::types::GenomeSketch, bincode::Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.length > 0 {
            use serde::Deserialize;
            let item = sylph::types::GenomeSketch::deserialize(&mut self.reader).unwrap();
            self.length -= 1;
            Some(Ok(item))
        } else {
            None
        }
    }
}

#[pyclass(module = "pysylph.lib")]
pub struct DatabaseFile {
    reader: DatabaseReader<std::io::BufReader<std::fs::File>>,
}

#[pymethods]
impl DatabaseFile {
    #[new]
    fn __new__(path: &str) -> PyResult<Self> {
        let f = std::fs::File::open(path).map(std::io::BufReader::new)?;
        let reader = DatabaseReader::new(f).unwrap();
        Ok(Self { reader })
    }

    fn __len__(&self) -> usize {
        self.reader.length
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<Py<GenomeSketch>>> {
        let py = unsafe { Python::assume_gil_acquired() };
        match self.reader.next() {
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
            None => Ok(None),
            Some(Ok(item)) => Some(Py::new(
                py,
                PyClassInitializer::from(Sketch).add_subclass(GenomeSketch::from(item)),
            ))
            .transpose(),
        }
    }

    // fn read_all(&mut self) -> Database {
    //     let mut db = Database::from(Vec::with_capacity(self.reader.length));
    //     for sketch in &mut self.reader {
    //         db.sketches.push(Arc::new(sketch.unwrap()));
    //     }
    //     let vec = Vec::<Arc<sylph::types::GenomeSketch>>::deserialize(&mut self.reader.reader).unwrap();
    //     Database::from(vec)
    // }

    // fn read(&mut self) -> GenomeSketch {
    //     use serde::de::Deserialize;
    //     let l: usize = serde::Deserialize::deserialize(&mut d).unwrap();
    //     // println!("{:?}", d);
    //     unimplemented!()
    // }
}

// --- SampleSketch ----------------------------------------------------------

/// A (query) sample sketch.
#[pyclass(module = "pysylph.lib", frozen, extends=Sketch)]
pub struct SampleSketch {
    sketch: sylph::types::SequencesSketch,
}

impl From<sylph::types::SequencesSketch> for SampleSketch {
    fn from(sketch: sylph::types::SequencesSketch) -> Self {
        Self { sketch }
    }
}

impl From<sylph::types::SequencesSketchEncode> for SampleSketch {
    fn from(sketch: sylph::types::SequencesSketchEncode) -> Self {
        Self::from(SequencesSketch::from_enc(sketch))
    }
}

#[pymethods]
impl SampleSketch {
    pub fn __repr__<'py>(slf: PyRef<'py, Self>) -> PyResult<String> {
        Ok(format!("<SampleSketch name={:?}>", slf.sketch.file_name))
    }

    /// `int`: The subsampling rate this sketch was built with.
    #[getter]
    pub fn c(&self) -> usize {
        self.sketch.c
    }

    /// `int`: The k-mer size this sketch was built with.
    #[getter]
    pub fn k(&self) -> usize {
        self.sketch.k
    }

    /// Load a sequence sketch from a path.
    #[classmethod]
    #[pyo3(signature = (file))]
    fn load<'py>(cls: &Bound<'py, PyType>, file: &Bound<'py, PyAny>) -> PyResult<Py<Self>> {
        let py = cls.py();
        // attempt to extract a path, or fall back to reader
        let result: Result<sylph::types::SequencesSketchEncode, _> = if let Ok(s) = py
            .import_bound(pyo3::intern!(py, "os"))?
            .call_method1(pyo3::intern!(py, "fsdecode"), (file,))
        {
            let path = s.downcast::<PyString>()?;
            let f = std::fs::File::open(path.to_str()?)?;
            let reader = std::io::BufReader::new(f);
            bincode::deserialize_from(reader)
        } else {
            let f = self::pyfile::PyFileRead::from_ref(&file)?;
            let reader = std::io::BufReader::new(f);
            bincode::deserialize_from(reader)
        };
        // handle error
        match result {
            Ok(sketch) => Py::new(
                py,
                PyClassInitializer::from(Sketch).add_subclass(Self::from(sketch)),
            ),
            Err(e) => match *e {
                bincode::ErrorKind::Io(io) => Err(io.into()),
                bincode::ErrorKind::InvalidUtf8Encoding(e) => Err(e.into()),
                other => Err(PyValueError::new_err(format!(
                    "failed to load db: {:?}",
                    other
                ))),
            },
        }
    }

    /// Dump a sequence sketch to a path.
    fn dump<'py>(slf: PyRef<'py, Self>, path: PyBackedStr) -> PyResult<()> {
        let f = std::fs::File::create(&*path)?;

        bincode::serialize_into(f, &slf.sketch).unwrap();
        Ok(())
    }
}

// --- ANIResult ---------------------------------------------------------------

/// An ANI result.
#[pyclass(module = "pysylph.lib", frozen, subclass)]
pub struct AniResult {
    // FIXME: currently works because of shitty unsafe code, ultimately this
    //        struct should only extract the needed attributes from AniResult
    //        not to have to worry about reference lifetimes.
    result: sylph::types::AniResult<'static>,
    genome: Py<GenomeSketch>,
}

#[pymethods]
impl AniResult {
    pub fn __repr__<'py>(slf: PyRef<'py, Self>) -> PyResult<String> {
        let py = slf.py();
        let genome = slf.genome.bind(py).borrow();
        Ok(format!(
            "<AniResult genome={:?} ani={:?}>",
            genome.sketch.file_name, slf.result.final_est_ani
        ))
    }

    /// `~pysylph.GenomeSketch`: A reference to the genome sketch.
    #[getter]
    fn genome_sketch<'py>(slf: PyRef<'py, Self>) -> Py<GenomeSketch> {
        let py = slf.py();
        slf.genome.clone_ref(py)
    }

    /// `float`: The coverage-corrected containment ANI, as a percentage.
    #[getter]
    fn ani<'py>(slf: PyRef<'py, Self>) -> f64 {
        f64::min(slf.result.final_est_ani * 100.0, 100.0)
    }

    /// `float`: The containment ANI without adjustment, as a percentage.
    #[getter]
    fn ani_naive<'py>(slf: PyRef<'py, Self>) -> f64 {
        slf.result.naive_ani * 100.0
    }

    /// `float`: An estimate of the effective, or the true coverage.
    #[getter]
    fn coverage<'py>(slf: PyRef<'py, Self>) -> f64 {
        slf.result.final_est_cov
    }
}

#[pyclass(module = "pysylph.lib", frozen, extends=AniResult)]
pub struct ProfileResult {}

#[pymethods]
impl ProfileResult {
    pub fn __repr__<'py>(slf: PyRef<'py, Self>) -> PyResult<String> {
        let py = slf.py();
        let genome = slf.as_super().genome.bind(py).borrow();
        Ok(format!(
            "<ProfileResult genome={:?} ani={:?}>",
            genome.sketch.file_name,
            slf.as_super().result.final_est_ani
        ))
    }

    /// `float`: The normalized sequence abundance, as a percentage.
    ///
    /// This is the *percentage of reads* assigned to each genome, similar
    /// to how Kraken computes abundance.
    #[getter]
    fn sequence_abundance<'py>(slf: PyRef<'py, Self>) -> f64 {
        *slf.as_super()
            .result
            .seq_abund
            .as_ref()
            .expect("ProfileResult should always have a sequence abundance set")
    }

    /// `float`: The normalized taxonomic abundance, as a percentage. Coverage-normalized - same as MetaPhlAn abundance
    ///
    /// This is the *coverage-normalized* abundanced, similar to how
    /// MetaPhlAn computes abundance.
    #[getter]
    fn taxonomic_abundance<'py>(slf: PyRef<'py, Self>) -> f64 {
        *slf.as_super()
            .result
            .rel_abund
            .as_ref()
            .expect("ProfileResult should always have a taxonomic abundance set")
    }

    /// `int`: The number of k-mers reassigned away from the genome.
    #[getter]
    fn kmers_reassigned<'py>(slf: PyRef<'py, Self>) -> usize {
        *slf.as_super()
            .result
            .kmers_lost
            .as_ref()
            .expect("ProfileResult should always have reassigned kmers set")
    }
}

// --- Sketcher ----------------------------------------------------------------

/// A ``sylph`` sketcher.
#[pyclass(module = "pysylph.lib", frozen)]
pub struct Sketcher {
    c: usize,
    k: usize,
    min_spacing: usize,
}

#[pymethods]
impl Sketcher {
    #[new]
    #[pyo3(signature = (c = 200, k = 31))]
    pub fn __new__(c: usize, k: usize) -> PyResult<Self> {
        if k != 21 && k != 31 {
            return Err(PyValueError::new_err(format!(
                "invalid k: expected 21 or 31, got {}",
                k
            )));
        }
        Ok(Self {
            c,
            k,
            min_spacing: 30,
        })
    }

    #[pyo3(signature = (name, contigs, pseudotax=true))]
    fn sketch_genome<'py>(
        slf: PyRef<'py, Self>,
        name: String,
        contigs: Bound<'py, PyAny>,
        pseudotax: bool,
    ) -> PyResult<Py<GenomeSketch>> {
        let py = slf.py();

        let mut gsketch = sylph::types::GenomeSketch::default();
        gsketch.min_spacing = slf.min_spacing;
        gsketch.c = slf.c;
        gsketch.k = slf.k;
        gsketch.file_name = name;
        if pseudotax {
            gsketch.pseudotax_tracked_nonused_kmers = Some(Vec::new());
        }

        // extract records
        let sequences = contigs
            .iter()?
            .map(|r| r.and_then(|s| PyBackedStr::extract_bound(&s)))
            .collect::<PyResult<Vec<_>>>()?;

        // sketch all records while allowing parallel code
        py.allow_threads(|| {
            // extract candidate kmers
            let mut markers = Vec::new();
            for (index, sequence) in sequences.iter().enumerate() {
                sylph::sketch::extract_markers_positions(
                    sequence.as_bytes(),
                    &mut markers,
                    gsketch.c,
                    gsketch.k,
                    index,
                );
                gsketch.gn_size += sequence.as_bytes().len();
            }

            // split duplicate / unique kmers
            let mut kmer_set = sylph::types::MMHashSet::default();
            let mut duplicate_set = sylph::types::MMHashSet::default();
            markers.sort(); // NB(@althonos): is this necessary here?
            for (_, _, km) in markers.iter() {
                if !kmer_set.insert(km) {
                    duplicate_set.insert(km);
                }
            }

            // record kmers
            let mut last_pos = 0;
            let mut last_contig = 0;
            for &(contig, pos, km) in markers.iter() {
                if !duplicate_set.contains(&km) {
                    if last_pos == 0
                        || last_contig != contig
                        || pos > gsketch.min_spacing + last_pos
                    {
                        gsketch.genome_kmers.push(km);
                        last_contig = contig;
                        last_pos = pos;
                    } else if let Some(kmers) = &mut gsketch.pseudotax_tracked_nonused_kmers {
                        kmers.push(km);
                    }
                }
            }
        });

        Py::new(
            py,
            PyClassInitializer::from(Sketch).add_subclass(GenomeSketch::from(gsketch)),
        )
    }

    #[pyo3(signature = (name, reads))]
    fn sketch_single<'py>(
        slf: PyRef<'py, Self>,
        name: String,
        reads: Bound<'py, PyAny>,
    ) -> PyResult<Py<SampleSketch>> {
        let py = slf.py();

        let mut kmer_map = std::collections::HashMap::default();
        // let ref_file = &read_file;
        // let reader = parse_fastx_file(&ref_file);
        let mut mean_read_length = 0.;
        let mut counter = 0usize;
        let mut kmer_to_pair_table = fxhash::FxHashSet::default();
        let mut num_dup_removed = 0;

        for result in reads.iter()? {
            let read: PyBackedStr = result?.extract()?;
            let seq = read.as_bytes();

            let mut vec = vec![];
            let kmer_pair = if seq.len() > 0 {
                None
            } else {
                self::exports::sketch::pair_kmer_single(seq)
            };
            sylph::sketch::extract_markers(&seq, &mut vec, slf.c, slf.k);
            for km in vec {
                self::exports::sketch::dup_removal_lsh_full_exact(
                    &mut kmer_map,
                    &mut kmer_to_pair_table,
                    &km,
                    kmer_pair,
                    &mut num_dup_removed,
                    false, //no_dedup,
                    Some(sylph::constants::MAX_DEDUP_COUNT),
                );
            }
            //moving average
            counter += 1;
            mean_read_length += ((seq.len() as f64) - mean_read_length) / counter as f64;
        }

        let sketch = sylph::types::SequencesSketch {
            kmer_counts: kmer_map,
            file_name: name,
            c: slf.c,
            k: slf.k,
            paired: false,
            sample_name: None,
            mean_read_length,
        };

        Py::new(
            py,
            PyClassInitializer::new(SampleSketch::from(sketch), PyClassInitializer::from(Sketch)),
        )
    }
}

// --- Functions ---------------------------------------------------------------

/// A ``sylph`` profiler.
#[pyclass(module = "pysylph.lib", frozen)]
pub struct Profiler {
    minimum_ani: Option<f64>,
    seq_id: Option<f64>,
    estimate_unknown: bool,
    min_number_kmers: usize,
    #[pyo3(get)]
    database: Py<Database>,
}

#[pymethods]
impl Profiler {
    #[new]
    #[pyo3(signature = (database, *, minimum_ani = None, seq_id = None, estimate_unknown = false, min_number_kmers = 50))]
    pub fn __new__<'py>(
        database: Py<Database>,
        minimum_ani: Option<f64>,
        seq_id: Option<f64>,
        estimate_unknown: bool,
        min_number_kmers: usize,
    ) -> PyResult<Self> {
        if let Some(m) = minimum_ani {
            if m < 0.0 || m > 100.0 {
                return Err(PyValueError::new_err(format!(
                    "invalid value for minimum_ani: {}",
                    m
                )));
            }
        }
        Ok(Self {
            minimum_ani,
            database,
            estimate_unknown,
            seq_id,
            min_number_kmers,
        })
    }

    #[pyo3(signature = (sample))]
    fn query<'py>(&self, sample: PyRef<'py, SampleSketch>) -> PyResult<Vec<Py<AniResult>>> {
        let py = sample.py();
        let database = self.database.bind(py).borrow();
        let args = sylph::cmdline::ContainArgs {
            minimum_ani: self.minimum_ani,
            files: Default::default(),
            file_list: Default::default(),
            min_count_correct: 3.0,
            min_number_kmers: self.min_number_kmers as f64,
            threads: 3,
            sample_threads: None,
            trace: false,
            debug: false,
            estimate_unknown: self.estimate_unknown,
            seq_id: self.seq_id,
            redundant_ani: 99.0,
            first_pair: Default::default(),
            second_pair: Default::default(),
            c: 200,
            k: 3,
            individual: false,
            min_spacing_kmer: 30,
            out_file_name: None,
            log_reassignments: false,
            pseudotax: false,
            ratio: false,
            mme: false,
            mle: false,
            nb: false,
            no_ci: false,
            no_adj: false,
            mean_coverage: false,
        };

        // estimate sample kmer identity
        let kmer_id_opt = if let Some(x) = args.seq_id {
            Some(x.powf(sample.sketch.k as f64))
        } else {
            self::exports::contain::get_kmer_identity(&sample.sketch, args.estimate_unknown)
        };

        // extract all matching kmers
        let sample_sketch = &sample.sketch;
        let database_sketches = &database.sketches;
        let mut stats = py.allow_threads(|| {
            database_sketches
                .par_iter()
                .flat_map_iter(|sketch| {
                    self::exports::contain::get_stats(&args, &sketch, &sample_sketch, None, false)
                })
                .collect::<Vec<_>>()
        });

        // estimate true coverage
        self::exports::contain::estimate_true_cov(
            &mut stats,
            kmer_id_opt,
            args.estimate_unknown,
            sample.sketch.mean_read_length,
            sample.sketch.k,
        );

        // sort by ANI
        // if pseudotax {} else {
        stats.sort_by(|x, y| y.final_est_ani.partial_cmp(&x.final_est_ani).unwrap());
        // }

        // Ok(())
        stats
            .into_iter()
            .map(|r| {
                let sketch = database
                    .sketches
                    .iter()
                    .find(|x| r.genome_sketch.file_name == x.file_name)
                    .unwrap();
                let genome = Py::new(
                    py,
                    PyClassInitializer::from(Sketch)
                        .add_subclass(GenomeSketch::from(sketch.clone())),
                )?;
                Py::new(
                    py,
                    PyClassInitializer::from(AniResult {
                        result: unsafe { std::mem::transmute(r) },
                        genome,
                    }),
                )
            })
            .collect()
    }

    #[pyo3(signature = (sample))]
    fn profile<'py>(&self, sample: PyRef<'py, SampleSketch>) -> PyResult<Vec<Py<ProfileResult>>> {
        let py = sample.py();
        let database = self.database.bind(py).borrow();
        let args = sylph::cmdline::ContainArgs {
            minimum_ani: self.minimum_ani,
            files: Default::default(),
            file_list: Default::default(),
            min_count_correct: 3.0,
            min_number_kmers: self.min_number_kmers as f64,
            threads: 3,
            sample_threads: None,
            trace: false,
            debug: false,
            estimate_unknown: self.estimate_unknown,
            seq_id: self.seq_id,
            redundant_ani: 99.0,
            first_pair: Default::default(),
            second_pair: Default::default(),
            c: 200,
            k: 3,
            individual: false,
            min_spacing_kmer: 30,
            out_file_name: None,
            log_reassignments: false,
            pseudotax: true,
            ratio: false,
            mme: false,
            mle: false,
            nb: false,
            no_ci: false,
            no_adj: false,
            mean_coverage: false,
        };

        // estimate sample kmer identity
        let kmer_id_opt = if let Some(x) = args.seq_id {
            Some(x.powf(sample.sketch.k as f64))
        } else {
            self::exports::contain::get_kmer_identity(&sample.sketch, args.estimate_unknown)
        };

        // extract all matching kmers
        let sample_sketch = &sample.sketch;
        let database_sketches = &database.sketches;
        let mut stats = py.allow_threads(|| {
            database_sketches
                .par_iter()
                .flat_map_iter(|sketch| {
                    self::exports::contain::get_stats(&args, &sketch, &sample_sketch, None, false)
                })
                .collect::<Vec<_>>()
        });

        // estimate true coverage
        self::exports::contain::estimate_true_cov(
            &mut stats,
            kmer_id_opt,
            args.estimate_unknown,
            sample.sketch.mean_read_length,
            sample.sketch.k,
        );

        if true {
            // reassign k-mers
            let winner_map = self::exports::contain::winner_table(&stats, false);
            let remaining_stats = stats
                .iter()
                .map(|x| x.genome_sketch)
                .flat_map(|genome_sketch| {
                    self::exports::contain::get_stats(
                        &args,
                        &genome_sketch,
                        &sample.sketch,
                        Some(&winner_map),
                        false,
                    )
                })
                .collect();
            stats = self::exports::contain::derep_if_reassign_threshold(
                &stats,
                remaining_stats,
                args.redundant_ani,
                sample.sketch.k,
            );
            self::exports::contain::estimate_true_cov(
                &mut stats,
                kmer_id_opt,
                args.estimate_unknown,
                sample.sketch.mean_read_length,
                sample.sketch.k,
            );

            // estimate unknown if needed
            let mut bases_explained = 1.;
            if args.estimate_unknown {
                bases_explained = self::exports::contain::estimate_covered_bases(
                    &stats,
                    &sample.sketch,
                    sample.sketch.mean_read_length,
                    sample.sketch.k,
                );
            }

            // compute coverage and relative abundances
            let total_cov = stats.iter().map(|x| x.final_est_cov).sum::<f64>();
            let total_seq_cov = stats
                .iter()
                .map(|x| x.final_est_cov * x.genome_sketch.gn_size as f64)
                .sum::<f64>();
            for result in stats.iter_mut() {
                result.rel_abund = Some(result.final_est_cov / total_cov * 100.0);
                result.seq_abund = Some(
                    result.final_est_cov * result.genome_sketch.gn_size as f64 / total_seq_cov
                        * 100.0
                        * bases_explained,
                );
            }
        }

        // sort by relative abundance
        stats.sort_by(|x, y| {
            y.rel_abund
                .unwrap()
                .partial_cmp(&x.rel_abund.unwrap())
                .unwrap()
        });

        // convert result
        stats
            .into_iter()
            .map(|r| {
                let sketch = database
                    .sketches
                    .iter()
                    .find(|x| r.genome_sketch.file_name == x.file_name)
                    .unwrap();
                let genome = Py::new(
                    py,
                    PyClassInitializer::from(Sketch)
                        .add_subclass(GenomeSketch::from(sketch.clone())),
                )?;
                Py::new(
                    py,
                    PyClassInitializer::from(AniResult {
                        result: unsafe { std::mem::transmute(r) },
                        genome,
                    })
                    .add_subclass(ProfileResult {}),
                )
            })
            .collect()
    }
}

// --- Initializer -------------------------------------------------------------

/// PyO3 bindings to ``sylph``, an ultrafast taxonomic profiler.
#[pymodule]
#[pyo3(name = "lib")]
pub fn init(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    m.add("__package__", "pysylph")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", env!("CARGO_PKG_AUTHORS").replace(':', "\n"))?;

    m.add_class::<Database>()?;
    m.add_class::<DatabaseFile>()?;

    m.add_class::<Sketch>()?;
    m.add_class::<GenomeSketch>()?;
    m.add_class::<SampleSketch>()?;

    m.add_class::<Sketcher>()?;
    m.add_class::<Profiler>()?;

    m.add_class::<AniResult>()?;

    Ok(())
}
