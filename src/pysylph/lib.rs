extern crate bincode;
extern crate pyo3;
extern crate sylph;
extern crate memmap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyTuple;
use pyo3::types::PyList;

// --- GenomeSketch ------------------------------------------------------------

/// A (reference) genome sketch.
#[pyclass(module = "pysylph.lib", frozen)]
pub struct GenomeSketch {
    sketch: sylph::types::GenomeSketch,
}

impl From<sylph::types::GenomeSketch> for GenomeSketch {
    fn from(sketch: sylph::types::GenomeSketch) -> Self {
        Self { sketch }
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
    genome: Py<GenomeSketch>,
}

#[pymethods]
impl AniResult {
    pub fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        Ok(format!("<AniResult {:?}>", self.result))
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
pub fn load_syldb(file: &str) -> PyResult<Vec<GenomeSketch>> {
    let f = std::fs::File::open(file)
        .unwrap();
    let m = unsafe {
        memmap::Mmap::map(&f).expect("failed to memmap")
    };
    let result: Vec<sylph::types::GenomeSketch> = bincode::deserialize(&m)
        .map_err(|e| PyValueError::new_err(format!("failed to load db: {:?}", e)))?;
    Ok(result.into_iter().map(GenomeSketch::from).collect())
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
fn query<'py>(gs: PyRef<'py, GenomeSketch>, ss: PyRef<'py, SequenceSketch>) -> PyResult<()> {
    let py = gs.py();

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


    let res = sylph::contain::get_stats(
        &args, 
        &gs.sketch, 
        &ss.sketch, 
        None,
        false,
    ).expect("nope");

    // let r2 = unsafe { std::mem::transmute(res)};

    // Ok(AniResult {
    //     result: r2,
    //     genome: gs.into()
    // })

    {
        let print_final_ani = format!("{:.2}", f64::min(res.final_est_ani * 100., 100.));
        let lambda_print = match res.lambda {
            sylph::types::AdjustStatus::Lambda(l) => format!("{:.3}", l),
            sylph::types::AdjustStatus::High => format!("HIGH"),
            sylph::types::AdjustStatus::Low => format!("LOW"),
        };
       
        let low_ani = res.ani_ci.0;
        let high_ani = res.ani_ci.1;
        let low_lambda = res.lambda_ci.0;
        let high_lambda = res.lambda_ci.1;

        let ci_ani;
        if low_ani.is_none() || high_ani.is_none() {
            ci_ani = "NA-NA".to_string();
        } else {
            ci_ani = format!(
                "{:.2}-{:.2}",
                low_ani.unwrap() * 100.,
                high_ani.unwrap() * 100.
            );
        }

        let ci_lambda;
        if low_lambda.is_none() || high_lambda.is_none() {
            ci_lambda = "NA-NA".to_string();
        } else {
            ci_lambda = format!("{:.2}-{:.2}", low_lambda.unwrap(), high_lambda.unwrap());
        }
        println!(
            "{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{:.0}\t{:.3}\t{}/{}\t{:.2}\t{}",
            res.seq_name,
            res.gn_name,
            print_final_ani,
            res.final_est_cov,
            ci_ani,
            lambda_print,
            ci_lambda,
            res.median_cov,
            res.mean_cov,
            res.containment_index.0,
            res.containment_index.1,
            res.naive_ani * 100.,
            res.contig_name,
        );
    }

    Ok(())
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

    m.add_class::<GenomeSketch>()?;
    m.add_class::<SequenceSketch>()?;

    m.add_function(wrap_pyfunction!(load_syldb, &m)?)?;
    m.add_function(wrap_pyfunction!(load_sylsp, &m)?)?;
    m.add_function(wrap_pyfunction!(query, &m)?)?;

    Ok(())
}
