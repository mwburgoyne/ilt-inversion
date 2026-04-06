use pyo3::prelude::*;

mod gwr;
mod bessel;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gwr::gwr_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bessel::besseli_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bessel::besselk_rust, m)?)?;
    Ok(())
}
