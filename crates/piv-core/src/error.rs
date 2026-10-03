use thiserror::Error;

#[derive(Debug, Error)]
pub enum PivError {
    #[error("invalid frame rate {0:?}")]
    FrameRate(String),
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("frames have different shapes: {a:?} vs {b:?}")]
    ShapeMismatch {
        a: (usize, usize),
        b: (usize, usize),
    },
    #[error("image ({height}×{width}) is smaller than the {window} px interrogation window")]
    ImageTooSmall {
        height: usize,
        width: usize,
        window: usize,
    },
}
