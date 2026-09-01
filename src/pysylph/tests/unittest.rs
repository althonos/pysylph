extern crate pyo3;
extern crate pysylph;

use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyModule;
use pyo3::Python;

pub fn main() -> PyResult<()> {
    // get the relative path to the project folder
    let folder = Path::new(file!())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("src");

    // spawn a Python interpreter
    Python::initialize();
    Python::attach(|py| {
        // insert the project folder in `sys.modules` so that
        // the main module can be imported by Python
        let sys = py.import("sys")?;
        sys.getattr("path")?.cast::<PyList>()?.insert(0, folder)?;

        // create a Python module from our rust code with debug symbols
        let module = PyModule::new(py, "pysylph.lib")?;
        pysylph::init(py, module.clone()).unwrap();
        sys.getattr("modules")?
            .cast::<PyDict>()?
            .set_item("pysylph.lib", &module)?;

        // run unittest on the tests
        let kwargs = PyDict::new(py);
        kwargs.set_item("exit", false).unwrap();
        kwargs.set_item("verbosity", 2u8).unwrap();
        py.import("unittest").unwrap().call_method(
            "TestProgram",
            ("pysylph.tests",),
            Some(&kwargs),
        )?;

        Ok(())
    })
}
