//! Core planar image velocimetry (PIV) algorithms.
//!
//! The pipeline for one image pair is:
//!
//! 1. [`preprocess::PreparedFrame`]: optional background subtraction, Gaussian smoothing and
//!    local contrast normalization, plus integral images used for signal masking.
//! 2. [`multipass::PivAnalyzer`]: multi-pass FFT cross-correlation with continuous (sub-pixel)
//!    window offset and Gaussian sub-pixel peak fitting.
//! 3. [`validate`]: peak-ratio and normalized-median tests, replacement of outliers.
//!
//! [`smooth`] post-processes a time series of fields (temporal and spatial filtering).
//!
//! All displacements are in pixels; converting to velocity is a matter of dividing by the
//! frame separation from [`frame::FrameRate::dt`].
//!
//! This crate has no I/O dependencies; decoding and output live in `piv-io`.

pub mod correlate;
pub mod error;
pub mod field;
pub mod frame;
pub mod grid;
pub mod multipass;
pub mod preprocess;
pub mod smooth;
pub mod synthetic;
pub mod validate;

pub use error::PivError;
pub use field::{VectorField, flags};
pub use frame::{Frame, FrameRate};
pub use grid::Grid;
pub use multipass::{MaskConfig, PivAnalyzer, PivConfig};
pub use preprocess::{PreparedFrame, PreprocessConfig};
pub use validate::ValidationConfig;

pub type Result<T> = std::result::Result<T, PivError>;
