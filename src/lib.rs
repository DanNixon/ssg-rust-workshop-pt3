use hdf5::{Dataset, File};
use ndarray::{Array1, ArrayBase, Data, Ix1, OwnedRepr, RawData, RawDataClone};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub struct Histogram<S: RawData<Elem = f64> + RawDataClone + Data> {
    pub edges: ArrayBase<S, Ix1>,
    pub bins: ArrayBase<S, Ix1>,
}

impl Histogram<OwnedRepr<f64>> {
    fn from_hdf5(edges: &Dataset, bins: &Dataset) -> hdf5::Result<Self> {
        let edges: Array1<f64> = edges.read_1d()?;
        let bins: Array1<f64> = bins.read_1d()?;
        Ok(Histogram { edges, bins })
    }

    pub fn load(filename: &str) -> hdf5::Result<Self> {
        let file = File::open(filename)?;
        let edges = file.dataset("/histogram/edges")?;
        let bins = file.dataset("/histogram/bins")?;
        Self::from_hdf5(&edges, &bins)
    }
}

pub fn integrate<S: RawData<Elem = f64> + RawDataClone + Data>(
    hist: Histogram<S>,
    range: (f64, f64),
) -> f64 {
    if hist.edges.is_empty() {
        return 0.0;
    }

    if hist.edges.len() != hist.bins.len() + 1 {
        panic!("Edges must have one more elements than bins");
    }

    if range.0 > range.1 {
        panic!("Range lower bound should not be greater than upper bound");
    }

    let factors: Array1<f64> = hist
        .edges
        .windows(2)
        .into_iter()
        .map(|window| {
            let (bin_lower, bin_upper) = (window[0], window[1]);
            let bin_width = bin_upper - bin_lower;

            if range.0 <= bin_lower && bin_upper <= range.1 {
                1.0
            } else if bin_lower <= range.0 && range.0 <= bin_upper {
                (bin_upper - range.0) / bin_width
            } else if bin_lower <= range.1 && range.1 <= bin_upper {
                (range.1 - bin_lower) / bin_width
            } else {
                0.0
            }
        })
        .collect();

    (&hist.bins * factors).sum()
}

#[pyclass]
pub struct OwnedHisto {
    inner: Histogram<OwnedRepr<f64>>,
}

#[pymethods]
impl OwnedHisto {
    #[staticmethod]
    fn load(filename: &str) -> Self {
        Self {
            inner: Histogram::load(filename).expect("Failed to load histogram from HDF5 file"),
        }
    }

    fn integrate(&self, range: (f64, f64)) -> PyResult<f64> {
        Ok(integrate(self.inner.clone(), range))
    }
}

#[pymodule]
fn rust_workshop_3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OwnedHisto>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn integrate_no_data() {
        let hist = Histogram {
            edges: array![],
            bins: array![],
        };
        let range = (0.0, 0.0);
        assert_eq!(integrate(hist, range), 0.0);
    }

    #[test]
    fn integrate_entire_range() {
        let hist = Histogram {
            edges: array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            bins: array![-1.0, 0.0, 1.0, 2.0, 3.0],
        };
        let range = (-1.0, 6.0);
        assert_eq!(integrate(hist, range), 5.0);
    }

    #[test]
    #[should_panic]
    fn integrate_bad_range() {
        let hist = Histogram {
            edges: array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            bins: array![-1.0, 0.0, 1.0, 2.0, 3.0],
        };
        let range = (12.0, 6.0);
        let _ = integrate(hist, range);
    }

    #[test]
    #[should_panic]
    fn integrate_bad_data_len() {
        let hist = Histogram {
            edges: array![0.0, 1.0, 2.0, 3.0, 4.0],
            bins: array![-1.0, 0.0, 1.0, 2.0, 3.0],
        };
        let range = (-1.0, 6.0);
        let _ = integrate(hist, range);
    }

    #[test]
    fn integrate_zero_range() {
        let hist = Histogram {
            edges: array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            bins: array![-1.0, 0.0, 1.0, 2.0, 3.0],
        };
        let range = (6.0, 6.0);
        assert_eq!(integrate(hist, range), 0.0);
    }

    #[test]
    fn integrate_partial_bin_coverage_lower_side_of_range() {
        let hist = Histogram {
            edges: array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            bins: array![-1.0, 0.0, 1.0, 2.0, 3.0],
        };
        let range = (2.75, 6.0);
        assert_eq!(integrate(hist, range), 5.25);
    }

    #[test]
    fn integrate_partial_bin_coverage_upper_side_of_range() {
        let hist = Histogram {
            edges: array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            bins: array![-1.0, 0.0, 1.0, 2.0, 4.0],
        };
        let range = (-1.0, 4.25);
        assert_eq!(integrate(hist, range), 3.0);
    }
}
