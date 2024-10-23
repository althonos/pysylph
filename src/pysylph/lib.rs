extern crate bincode;
extern crate pyo3;
extern crate sylph;
extern crate memmap;

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyTuple;
use pyo3::types::PyList;
use pyo3::types::PyType;

// --- GenomeSketch ------------------------------------------------------------

/// A (reference) genome sketch.
#[pyclass(module = "pysylph.lib", frozen)]
pub struct GenomeSketch {
    sketch: Arc<sylph::types::GenomeSketch>,
}

impl From<Arc<sylph::types::GenomeSketch>> for GenomeSketch {
    fn from(sketch: Arc<sylph::types::GenomeSketch>) -> Self {
        Self { 
            sketch
        }
    }
}

impl From<sylph::types::GenomeSketch> for GenomeSketch {
    fn from(sketch: sylph::types::GenomeSketch) -> Self {
        Self::from(Arc::new(sketch))
    }
}

#[pymethods]
impl GenomeSketch {
    pub fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        Ok(format!("<GenomeSketch name={:?}>", self.sketch.file_name))
    }

    #[getter]
    pub fn genome_size(&self) -> usize {
        self.sketch.gn_size
    }

    #[getter]
    pub fn c(&self) -> usize {
        self.sketch.c
    }

    #[getter]
    pub fn k(&self) -> usize {
        self.sketch.k
    }

    #[getter]
    pub fn kmers<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, &self.sketch.genome_kmers)
    }
}

// --- Database ----------------------------------------------------------------

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

    pub fn __getitem__<'py>(slf: PyRef<'py, Self>, item: isize) -> PyResult<GenomeSketch> {

        let mut item_ = item;
        if item_ < 0 {
            item_ += slf.sketches.len() as isize;
        }

        if item_ < 0 || item_ >= slf.sketches.len() as isize {
            Err(PyIndexError::new_err(item))
        } else {
            Ok(GenomeSketch::from(slf.sketches[item_ as usize].clone()))
        }
    }

    /// Load a database from a path.
    #[classmethod]
    #[pyo3(signature = (path, memmap = true))]
    fn load<'py>(cls: &Bound<'_, PyType>, path: PyBackedStr, memmap: bool) -> PyResult<Self> {
        // FIXME(@althonos): Add support for reading from a file-like object.

        // load file using either memmap or direct read
        let f = std::fs::File::open(&*path)?;
        let result: bincode::Result<Vec<Arc<sylph::types::GenomeSketch>>> = if memmap {
            let m = unsafe { memmap::Mmap::map(&f)? };
            bincode::deserialize(&m)
        } else {
            let reader = std::io::BufReader::new(f);
            bincode::deserialize_from(reader)
        };

        // hadnle error
        match result {
            Ok(sketches) => Ok(Database::from(sketches)),
            Err(e) => {
                match *e {
                    bincode::ErrorKind::Io(io) => Err(io.into()),
                    bincode::ErrorKind::InvalidUtf8Encoding(e) => Err(e.into()),
                    other => {
                        Err(PyValueError::new_err(format!("failed to load db: {:?}", other)))
                    }
                }
            }
        }
    }

    /// Dump a database to a path.
    fn dump<'py>(slf: PyRef<'py, Self>, path: PyBackedStr) -> PyResult<()> {
        let f = std::fs::File::create(&*path)?;

        bincode::serialize_into(f, &slf.sketches).unwrap();
        Ok(())
    }
}

// --- SequenceSketch ----------------------------------------------------------

/// A (query) sequence sketch .
#[pyclass(module = "pysylph.lib", frozen)]
pub struct SequenceSketch {
    sketch: sylph::types::SequencesSketch,
}

impl From<sylph::types::SequencesSketch> for SequenceSketch {
    fn from(sketch: sylph::types::SequencesSketch) -> Self {
        Self { sketch }
    }
}

#[pymethods]
impl SequenceSketch {
    pub fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        Ok(format!("<SequenceSketch name={:?}>", self.sketch.file_name))
    }
}

// --- ANIResult ---------------------------------------------------------------

/// An ANI result.
#[pyclass(module = "pysylph.lib", frozen)]
pub struct AniResult {
    result: sylph::types::AniResult<'static>,
    genome: Arc<sylph::types::GenomeSketch>,
}

#[pymethods]
impl AniResult {
    pub fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        Ok(format!("<AniResult genome={:?} ani={:?}>", self.genome.file_name, self.result.final_est_ani))
    }

    #[getter]
    fn ani<'py>(slf: PyRef<'py, Self>) -> f64 {
        f64::min(slf.result.final_est_ani, 1.0)
    }

    #[getter]
    fn ani_naive<'py>(slf: PyRef<'py, Self>) -> f64 {
        slf.result.naive_ani
    }

    #[getter]
    fn coverage<'py>(slf: PyRef<'py, Self>) -> f64 {
        slf.result.final_est_cov
    } 
}

// --- Sketcher ----------------------------------------------------------------

/// A genome sketcher.
#[pyclass]
pub struct GenomeSketcher {
    c: usize,
    k: usize,
    min_spacing: usize,
}

#[pymethods]
impl GenomeSketcher {
    #[new]
    #[pyo3(signature = (c = 200, k = 31))]
    pub fn __new__(c: usize, k: usize) -> PyResult<GenomeSketcher> {
        Ok(GenomeSketcher { c, k, min_spacing: 30 })
    }

    #[pyo3(signature = (name, sequence, *sequences))]
    fn sketch<'py>(&self, name: String, sequence: Bound<'py, PyAny>, sequences: Bound<'py, PyTuple>) -> PyResult<GenomeSketch> {
        let mut gsketch = sylph::types::GenomeSketch::default();
        gsketch.min_spacing = self.min_spacing;
        gsketch.c = self.c;
        gsketch.k = self.k;
        gsketch.file_name = name;

        // extract candidate kmers
        let mut markers = Vec::new();
        for (contig_index, sequence) in [sequence].into_iter().chain(sequences.iter()).enumerate() {
            let s = PyBackedStr::extract_bound(&sequence)?;
            sylph::sketch::extract_markers_positions(
                s.as_bytes(),
                &mut markers,
                self.c,
                self.k,
                contig_index,
            );
            gsketch.gn_size += s.as_bytes().len();
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
        
        //
        let mut last_pos = 0;
        let mut last_contig = 0;
        for &(contig, pos, km) in markers.iter() {
            if !duplicate_set.contains(&km) {
                if last_pos == 0 || last_contig != contig || pos > self.min_spacing + last_pos {
                    gsketch.genome_kmers.push(km);
                    last_contig = contig;
                    last_pos = pos;
                }
                //  else if pseudotax {
                //     pseudotax_track_kmers.push(*km);
                // }
            }
        }

        //
        Ok(GenomeSketch::from(gsketch))
    }
}

// --- Functions ---------------------------------------------------------------

#[pyfunction]
pub fn load_syldb(file: &str) -> PyResult<Database> {
    let f = std::fs::File::open(file)
        .unwrap();
    let m = unsafe {
        memmap::Mmap::map(&f).expect("failed to memmap")
    };
    let result: Vec<sylph::types::GenomeSketch> = bincode::deserialize(&m)
        .map_err(|e| PyValueError::new_err(format!("failed to load db: {:?}", e)))?;
    Ok(Database::from_iter(result.into_iter()))
}

#[pyfunction]
pub fn load_sylsp(file: &str) -> PyResult<SequenceSketch> {
    let f = std::fs::File::open(file)
        .unwrap();
    let m = unsafe {
        memmap::Mmap::map(&f).expect("failed to memmap")
    };
    let result: sylph::types::SequencesSketchEncode = bincode::deserialize(&m)
        .map_err(|e| PyValueError::new_err(format!("failed to load query: {:?}", e)))?;
    Ok(sylph::types::SequencesSketch::from_enc(result).into())
}

#[pyfunction]
#[pyo3(signature = (sample, database, seq_id = None, estimate_unknown = false))]
fn query<'py>(
    sample: PyRef<'py, SequenceSketch>, 
    database: PyRef<'py, Database>, 
    seq_id: Option<f64>,
    estimate_unknown: bool,
) -> PyResult<Vec<AniResult>> {
    let py = sample.py();

    let args = sylph::cmdline::ContainArgs {
        files: Default::default(),
        file_list: Default::default(),
        min_count_correct: 3.0,
        min_number_kmers: 60.0,
        minimum_ani: Some(0.9),
        threads: 3,
        sample_threads: None,
        trace: false,
        debug: false,
        estimate_unknown: false,
        seq_id: None,
        redundant_ani: 99.0,
        first_pair: Default::default(),
        second_pair: Default::default(),
        c: 200,    
        k: 31,
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
    let kmer_id_opt = if let Some(x) = seq_id {
        Some(x.powf(sample.sketch.k as f64))
    } else {
        sylph::contain::get_kmer_identity(&sample.sketch, estimate_unknown)
    };

    // extract all matching kmers
    let mut stats = Vec::new();
    for sketch in &database.sketches {
        if let Some(res) = sylph::contain::get_stats(
            &args, 
            &sketch, 
            &sample.sketch, 
            None,
            false,
        ) {
            stats.push(res);
        }
    }

    // estimate true coverage
    sylph::contain::estimate_true_cov(
        &mut stats, 
        kmer_id_opt, 
        estimate_unknown, 
        sample.sketch.mean_read_length, 
        sample.sketch.k
    );

    // sort by ANI
    // if pseudotax {} else {
    stats.sort_by(|x,y| y.final_est_ani.partial_cmp(&x.final_est_ani).unwrap());
    // }

    // Ok(())
    Ok(stats.into_iter()
        .map(|r| {
            let sketch = database.sketches.iter()
                .find(|x| r.genome_sketch.file_name == x.file_name )
                .unwrap();
            AniResult {
                result: unsafe { std::mem::transmute(r) },
                genome: sketch.clone(),
            }

        })
        .collect())
}


// --- Initializer -------------------------------------------------------------

/// PyO3 bindings to ``sylph``, an ultrafast taxonomic profiler.
#[pymodule]
#[pyo3(name = "lib")]
pub fn init(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    m.add("__package__", "pysylph")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", env!("CARGO_PKG_AUTHORS").replace(':', "\n"))?;

    m.add_class::<GenomeSketcher>()?;
    m.add_class::<Database>()?;

    m.add_class::<GenomeSketch>()?;
    m.add_class::<SequenceSketch>()?;

    m.add_function(wrap_pyfunction!(load_syldb, &m)?)?;
    m.add_function(wrap_pyfunction!(load_sylsp, &m)?)?;
    m.add_function(wrap_pyfunction!(query, &m)?)?;

    Ok(())
}
