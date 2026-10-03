//! I/O for piv: video decoding via FFmpeg, capture-rate discovery, physical scale, vector
//! field writers (HDF5 / NPZ) and quiver rendering.

pub mod decode;
pub mod metadata;
pub mod render;
#[cfg(feature = "hdf5")]
pub mod run_file;
pub mod scale;
pub mod write;

pub use decode::{Channel, DecodeOptions, FfmpegSource, FrameSource, Roi};
pub use metadata::{CaptureRate, MediaInfo, RateSource, probe, resolve_capture_rate};
pub use scale::{FovAxis, Scale, ScaleSpec};
