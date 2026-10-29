use std::path::PathBuf;
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use piv_io::render::RgbImage;

/// Encodes PNGs on a background thread; PNG encoding is slow relative to correlation.
pub struct PngSink {
    tx: SyncSender<(PathBuf, RgbImage)>,
    handle: JoinHandle<Result<usize>>,
}

impl PngSink {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::sync_channel::<(PathBuf, RgbImage)>(8);
        let handle = thread::spawn(move || -> Result<usize> {
            let mut n = 0;
            for (path, img) in rx {
                img.save(&path)
                    .with_context(|| format!("writing {}", path.display()))?;
                n += 1;
            }
            Ok(n)
        });
        Self { tx, handle }
    }

    pub fn send(&self, path: PathBuf, img: RgbImage) {
        // A send error means the writer stopped on an error, which `finish` reports.
        self.tx.send((path, img)).ok();
    }

    /// Waits for all queued images; returns how many were written.
    pub fn finish(self) -> Result<usize> {
        drop(self.tx);
        self.handle
            .join()
            .map_err(|_| anyhow::anyhow!("PNG writer panicked"))?
    }
}
