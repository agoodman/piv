use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Frame;

/// Image conditioning applied before correlation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessConfig {
    /// Gaussian smoothing sigma in pixels (0 disables). Suppresses sensor noise.
    pub smooth_sigma: f32,
    /// Half-size of the box used for local mean/std normalization (0 disables). Evens out
    /// brightness variation across the light sheet. Off by default: on real dry-ice fog
    /// footage a small radius amplified noise in low-contrast areas and made results much
    /// worse, although it helps on synthetic smooth texture.
    pub normalize_radius: usize,
    /// Added to the local std before dividing, so dark noisy regions are not amplified.
    pub normalize_eps: f32,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            smooth_sigma: 1.0,
            normalize_radius: 4,
            normalize_eps: 0.02,
        }
    }
}

/// How a static background image is estimated from sample frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackgroundMode {
    Min,
    Mean,
}

/// Accumulates a per-pixel background estimate over a set of frames.
#[derive(Debug, Clone)]
pub struct BackgroundEstimator {
    mode: BackgroundMode,
    acc: Option<Array2<f32>>,
    count: usize,
}

impl BackgroundEstimator {
    pub fn new(mode: BackgroundMode) -> Self {
        Self {
            mode,
            acc: None,
            count: 0,
        }
    }

    pub fn add(&mut self, frame: &Frame) {
        self.count += 1;
        match (&mut self.acc, self.mode) {
            (None, _) => self.acc = Some(frame.clone()),
            (Some(acc), BackgroundMode::Min) => acc.zip_mut_with(frame, |a, &b| *a = a.min(b)),
            (Some(acc), BackgroundMode::Mean) => *acc += frame,
        }
    }

    pub fn finish(self) -> Option<Frame> {
        let mut acc = self.acc?;
        if self.mode == BackgroundMode::Mean {
            acc /= self.count as f32;
        }
        Some(acc)
    }
}

/// Summed-area tables of intensity and squared intensity, for O(1) box statistics.
#[derive(Debug, Clone)]
pub struct IntegralImage {
    width: usize,
    height: usize,
    sum: Vec<f64>,
    sumsq: Vec<f64>,
}

impl IntegralImage {
    pub fn new(frame: &Frame) -> Self {
        let (height, width) = frame.dim();
        let stride = width + 1;
        let mut sum = vec![0.0; stride * (height + 1)];
        let mut sumsq = vec![0.0; stride * (height + 1)];
        for (y, row) in frame.rows().into_iter().enumerate() {
            let (mut rs, mut rq) = (0.0f64, 0.0f64);
            for (x, &v) in row.iter().enumerate() {
                let v = v as f64;
                rs += v;
                rq += v * v;
                let i = (y + 1) * stride + x + 1;
                sum[i] = sum[i - stride] + rs;
                sumsq[i] = sumsq[i - stride] + rq;
            }
        }
        Self {
            width,
            height,
            sum,
            sumsq,
        }
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    fn rect(table: &[f64], stride: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        table[y1 * stride + x1] - table[y0 * stride + x1] - table[y1 * stride + x0]
            + table[y0 * stride + x0]
    }

    /// Mean and standard deviation over the `w × h` box with top-left corner `(x, y)`.
    pub fn stats(&self, x: usize, y: usize, w: usize, h: usize) -> (f32, f32) {
        let s = self.width + 1;
        let n = (w * h) as f64;
        let mean = Self::rect(&self.sum, s, x, y, x + w, y + h) / n;
        let var = Self::rect(&self.sumsq, s, x, y, x + w, y + h) / n - mean * mean;
        (mean as f32, var.max(0.0).sqrt() as f32)
    }
}

/// A frame ready for correlation: the conditioned image plus statistics of the raw intensity
/// (after background subtraction) used to decide whether a window holds any signal.
#[derive(Debug, Clone)]
pub struct PreparedFrame {
    /// Conditioned image fed to the correlator.
    pub image: Frame,
    /// Integral image of the background-subtracted raw intensity.
    pub stats: IntegralImage,
}

impl PreparedFrame {
    pub fn new(raw: &Frame, cfg: &PreprocessConfig, background: Option<&Frame>) -> Self {
        let mut base = raw.clone();
        if let Some(bg) = background {
            base.zip_mut_with(bg, |v, &b| *v = (*v - b).max(0.0));
        }
        let stats = IntegralImage::new(&base);
        let mut image = if cfg.smooth_sigma > 0.0 {
            gaussian_blur(&base, cfg.smooth_sigma)
        } else {
            base
        };
        if cfg.normalize_radius > 0 {
            image = local_normalize(&image, cfg.normalize_radius, cfg.normalize_eps);
        }
        Self { image, stats }
    }

    pub fn dim(&self) -> (usize, usize) {
        self.image.dim()
    }
}

/// Separable Gaussian blur with edge clamping.
pub fn gaussian_blur(frame: &Frame, sigma: f32) -> Frame {
    let radius = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut kernel: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= total);

    let (h, w) = frame.dim();
    let src = frame.as_standard_layout();
    let src = src.as_slice().expect("standard layout");

    let mut tmp = vec![0.0f32; h * w];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &src[y * w..(y + 1) * w];
        for (x, o) in out.iter_mut().enumerate() {
            *o = kernel
                .iter()
                .enumerate()
                .map(|(k, &kv)| {
                    kv * row[(x as isize + k as isize - radius).clamp(0, w as isize - 1) as usize]
                })
                .sum();
        }
    });
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        for (k, &kv) in kernel.iter().enumerate() {
            let sy = (y as isize + k as isize - radius).clamp(0, h as isize - 1) as usize;
            let row = &tmp[sy * w..(sy + 1) * w];
            out.iter_mut().zip(row).for_each(|(o, &v)| *o += kv * v);
        }
    });
    Array2::from_shape_vec((h, w), out).expect("shape matches")
}

/// Subtract the local mean and divide by the local standard deviation (plus `eps`) over a
/// `(2r+1)²` box, clipped at the image edges.
pub fn local_normalize(frame: &Frame, radius: usize, eps: f32) -> Frame {
    let (h, w) = frame.dim();
    let ii = IntegralImage::new(frame);
    let src = frame.as_standard_layout();
    let src = src.as_slice().expect("standard layout");
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(h);
        for (x, o) in out.iter_mut().enumerate() {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(w);
            let (mean, std) = ii.stats(x0, y0, x1 - x0, y1 - y0);
            *o = (src[y * w + x] - mean) / (std + eps);
        }
    });
    Array2::from_shape_vec((h, w), out).expect("shape matches")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn integral_stats_match_direct() {
        let f = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        let ii = IntegralImage::new(&f);
        let (m, s) = ii.stats(1, 1, 2, 2);
        assert!((m - 7.25).abs() < 1e-6);
        let var = [5.0f32, 6.0, 8.0, 10.0]
            .iter()
            .map(|v| (v - 7.25).powi(2))
            .sum::<f32>()
            / 4.0;
        assert!((s - var.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn blur_preserves_constant_and_mass() {
        let f = Frame::from_elem((20, 30), 0.5);
        let b = gaussian_blur(&f, 1.5);
        assert!(b.iter().all(|&v| (v - 0.5).abs() < 1e-6));

        let mut d = Frame::zeros((21, 21));
        d[[10, 10]] = 1.0;
        let b = gaussian_blur(&d, 1.0);
        assert!((b.sum() - 1.0).abs() < 1e-5);
        assert!(b[[10, 10]] > b[[10, 11]] && b[[10, 11]] > b[[10, 12]]);
    }

    #[test]
    fn normalize_removes_gradient() {
        let f = Frame::from_shape_fn((40, 40), |(y, x)| 0.01 * x as f32 + 0.002 * y as f32);
        let n = local_normalize(&f, 5, 0.0);
        // A linear ramp is symmetric about the box centre, so the normalized interior is ~0.
        assert!(n[[20, 20]].abs() < 1e-3);
    }

    #[test]
    fn background_min_and_mean() {
        let a = array![[1.0, 4.0]];
        let b = array![[3.0, 2.0]];
        let mut e = BackgroundEstimator::new(BackgroundMode::Min);
        e.add(&a);
        e.add(&b);
        assert_eq!(e.finish().unwrap(), array![[1.0, 2.0]]);
        let mut e = BackgroundEstimator::new(BackgroundMode::Mean);
        e.add(&a);
        e.add(&b);
        assert_eq!(e.finish().unwrap(), array![[2.0, 3.0]]);
    }
}
