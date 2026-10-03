//! Multi-pass interrogation with continuous (sub-pixel) window offset.
//!
//! Each pass shifts the two windows symmetrically by the displacement predicted from the
//! previous pass, sampling them with bilinear interpolation, so the correlation only has to
//! measure a small residual. Repeating the final pass drives that residual toward zero, which
//! removes most of the bias toward zero displacement that smooth (fog-like) texture suffers.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::correlate::CorrelationPlan;
use crate::field::{VectorField, flags};
use crate::preprocess::PreparedFrame;
use crate::validate::{Limits, ValidationConfig, ValidationSummary, validate};
use crate::{Grid, PivError};

/// Thresholds on the (background-subtracted) raw intensity of a window, below which it is
/// treated as containing no seeding and is not correlated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskConfig {
    /// Minimum window mean intensity (0–1 scale).
    pub min_mean: f32,
    /// Minimum window intensity standard deviation (0–1 scale).
    pub min_std: f32,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            min_mean: 0.0,
            min_std: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PivConfig {
    /// Window size for each pass, largest first. Each must be even. The default `[96, 64]`
    /// suits smooth fog texture; sparse particle images resolve well with `[64, 32]`.
    pub windows: Vec<usize>,
    /// Fractional overlap between neighbouring windows, in `[0, 1)`.
    pub overlap: f32,
    /// How many times the final window size is run (≥ 1). Extra iterations refine the
    /// sub-pixel window offset.
    pub final_iterations: usize,
    pub mask: MaskConfig,
    pub validation: ValidationConfig,
}

impl Default for PivConfig {
    fn default() -> Self {
        Self {
            windows: vec![96, 64],
            overlap: 0.5,
            final_iterations: 2,
            mask: MaskConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

/// Result of analysing one pair: the final-pass field plus its validation summary.
#[derive(Debug, Clone)]
pub struct PairResult {
    pub field: VectorField,
    pub summary: ValidationSummary,
}

/// Runs the configured multi-pass analysis on prepared frame pairs. Construct once and reuse:
/// FFT plans are built up front.
pub struct PivAnalyzer {
    cfg: PivConfig,
    plans: HashMap<usize, Arc<CorrelationPlan>>,
}

impl PivAnalyzer {
    pub fn new(cfg: PivConfig) -> crate::Result<Self> {
        if cfg.windows.is_empty() {
            return Err(PivError::Config(
                "at least one window size is required".into(),
            ));
        }
        if let Some(w) = cfg.windows.iter().find(|&&w| w < 8 || w % 2 != 0) {
            return Err(PivError::Config(format!(
                "window size {w} must be even and ≥ 8"
            )));
        }
        if !(0.0..1.0).contains(&cfg.overlap) {
            return Err(PivError::Config(format!(
                "overlap {} must be in [0, 1)",
                cfg.overlap
            )));
        }
        if cfg.final_iterations == 0 {
            return Err(PivError::Config(
                "final_iterations must be at least 1".into(),
            ));
        }
        let plans = cfg
            .windows
            .iter()
            .map(|&w| (w, Arc::new(CorrelationPlan::new(w))))
            .collect();
        Ok(Self { cfg, plans })
    }

    pub fn config(&self) -> &PivConfig {
        &self.cfg
    }

    /// Grid of the final pass for an image of the given size.
    pub fn output_grid(&self, height: usize, width: usize) -> crate::Result<Grid> {
        Grid::new(
            height,
            width,
            *self.cfg.windows.last().expect("validated"),
            self.cfg.overlap,
        )
    }

    /// Displacement field from frame `a` to frame `b`, in pixels.
    pub fn analyze(&self, a: &PreparedFrame, b: &PreparedFrame) -> crate::Result<PairResult> {
        let (h, w) = a.dim();
        if b.dim() != (h, w) {
            return Err(PivError::ShapeMismatch {
                a: (h, w),
                b: b.dim(),
            });
        }
        let last = *self.cfg.windows.last().expect("validated");
        let passes = self
            .cfg
            .windows
            .iter()
            .copied()
            .chain(std::iter::repeat_n(last, self.cfg.final_iterations - 1));
        let mut result: Option<PairResult> = None;
        for win in passes {
            let grid = Grid::new(h, w, win, self.cfg.overlap)?;
            let mut field = self.pass(a, b, &grid, result.as_ref().map(|r| &r.field));
            let limits = Limits {
                max_displacement: self.cfg.windows[0] as f32 / 2.0,
            };
            let summary = validate(&mut field, &self.cfg.validation, limits);
            result = Some(PairResult { field, summary });
        }
        Ok(result.expect("at least one pass"))
    }

    fn pass(
        &self,
        a: &PreparedFrame,
        b: &PreparedFrame,
        grid: &Grid,
        predictor: Option<&VectorField>,
    ) -> VectorField {
        let (h, w) = a.dim();
        let n = grid.window;
        let plan = &self.plans[&n];
        let mask = &self.cfg.mask;
        // Fractional origins need one extra pixel for interpolation.
        let max_x = (w - n - 1) as f32;
        let max_y = (h - n - 1) as f32;

        let cells: Vec<(f32, f32, f32, u8)> = (0..grid.len())
            .into_par_iter()
            .map_init(
                || plan.workspace(),
                |ws, k| {
                    let (ix, iy) = (k % grid.nx, k / grid.nx);
                    let (x, y) = grid.window_origin(ix, iy);
                    let (px, py) = predictor.map_or((0.0, 0.0), |p| {
                        p.sample(grid.center_x(ix), grid.center_y(iy))
                    });
                    // Split the predicted shift symmetrically between the two windows so the
                    // vector stays located at the window centre. Near the border, move both
                    // windows inward together rather than shrinking the offset, so the
                    // iteration can still converge there.
                    let place = |origin: usize, shift: f32, max: f32| {
                        let shift = shift.clamp(-max, max);
                        let a0 = (origin as f32 - shift / 2.0)
                            .clamp((-shift).max(0.0), (max - shift).min(max));
                        (a0, a0 + shift)
                    };
                    let (ax, bx) = place(x, px, max_x);
                    let (ay, by) = place(y, py, max_y);

                    let has_signal = |f: &PreparedFrame, x: f32, y: f32| {
                        let (mean, std) =
                            f.stats.stats(x.round() as usize, y.round() as usize, n, n);
                        mean >= mask.min_mean && std >= mask.min_std
                    };
                    if !has_signal(a, ax, ay) || !has_signal(b, bx, by) {
                        return (f32::NAN, f32::NAN, 0.0, flags::NO_SIGNAL);
                    }
                    match plan.correlate(ws, &a.image, (ax, ay), &b.image, (bx, by)) {
                        Some(p) => (bx - ax + p.dx, by - ay + p.dy, p.ppr, 0),
                        None => (f32::NAN, f32::NAN, 0.0, flags::NO_SIGNAL),
                    }
                },
            )
            .collect();

        let mut field = VectorField::new(grid.clone());
        for (k, (dx, dy, ppr, fl)) in cells.into_iter().enumerate() {
            let idx = [k / grid.nx, k % grid.nx];
            field.dx[idx] = dx;
            field.dy[idx] = dy;
            field.peak_ratio[idx] = ppr;
            field.flags[idx] = fl;
        }
        field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::PreprocessConfig;
    use crate::synthetic;
    use crate::{Frame, field::flags};

    fn prep(a: &Frame, b: &Frame) -> (PreparedFrame, PreparedFrame) {
        let cfg = PreprocessConfig {
            smooth_sigma: 0.0,
            normalize_radius: 0,
            normalize_eps: 0.0,
        };
        (
            PreparedFrame::new(a, &cfg, None),
            PreparedFrame::new(b, &cfg, None),
        )
    }

    /// RMS error and worst error over vectors at least `margin` px from the image edge.
    fn errors(
        f: &VectorField,
        truth: impl Fn(f32, f32) -> (f32, f32),
        margin: f32,
        size: f32,
    ) -> (f32, f32, usize) {
        let (xs, ys) = (f.xs(), f.ys());
        let (mut sum, mut worst, mut n) = (0.0f32, 0.0f32, 0usize);
        for (iy, &y) in ys.iter().enumerate() {
            for (ix, &x) in xs.iter().enumerate() {
                if x < margin || y < margin || x > size - margin || y > size - margin {
                    continue;
                }
                let (u, v) = truth(x, y);
                let e = ((f.dx[[iy, ix]] - u).powi(2) + (f.dy[[iy, ix]] - v).powi(2)).sqrt();
                sum += e * e;
                worst = worst.max(e);
                n += 1;
            }
        }
        ((sum / n as f32).sqrt(), worst, n)
    }

    fn run(a: &Frame, b: &Frame, windows: Vec<usize>) -> PairResult {
        let (pa, pb) = prep(a, b);
        let final_iterations = if windows.len() == 1 { 1 } else { 2 };
        let cfg = PivConfig {
            windows,
            final_iterations,
            ..Default::default()
        };
        PivAnalyzer::new(cfg).unwrap().analyze(&pa, &pb).unwrap()
    }

    #[test]
    fn uniform_shifts_particles() {
        for (i, &(u, v)) in [
            (0.25, 0.0),
            (0.0, 0.5),
            (3.7, 0.0),
            (3.7, -1.3),
            (-2.2, 2.6),
        ]
        .iter()
        .enumerate()
        {
            let (a, b) =
                synthetic::particle_pair(256, 256, 10 + i as u64, 0.04, 1.2, |_, _| (u, v));
            let r = run(&a, &b, vec![64, 32]);
            let (rms, worst, n) = errors(&r.field, |_, _| (u, v), 20.0, 256.0);
            assert!(n > 100);
            assert!(rms < 0.05, "shift ({u}, {v}): rms {rms} worst {worst}");
            // Edge windows lose particles; allow a couple of outliers there.
            assert!(
                r.summary.median_outliers * 50 <= r.field.grid.len(),
                "shift ({u}, {v}): {:?}",
                r.summary
            );
        }
    }

    #[test]
    fn uniform_shift_fog_texture_with_noise() {
        let (u, v) = (2.4, -1.7);
        let (mut a, mut b) = synthetic::fog_pair(256, 256, 4, |_, _| (u, v));
        synthetic::add_noise(&mut a, 0.01, 1);
        synthetic::add_noise(&mut b, 0.01, 2);
        // Smooth texture needs the default high-pass normalization to correlate well.
        let pre = PreprocessConfig::default();
        let (pa, pb) = (
            PreparedFrame::new(&a, &pre, None),
            PreparedFrame::new(&b, &pre, None),
        );
        let r = PivAnalyzer::new(PivConfig::default())
            .unwrap()
            .analyze(&pa, &pb)
            .unwrap();
        let (rms, worst, _) = errors(&r.field, |_, _| (u, v), 20.0, 256.0);
        assert!(rms < 0.1, "rms {rms} worst {worst} {:?}", r.summary);
    }

    #[test]
    fn large_shift_recovered_by_multipass() {
        let (u, v) = (20.0, -6.0);
        let (a, b) = synthetic::particle_pair(320, 320, 5, 0.04, 1.2, |_, _| (u, v));
        let r = run(&a, &b, vec![64, 32]);
        // Vectors near the edge lose their partner window; look at the interior.
        let (rms, worst, _) = errors(&r.field, |_, _| (u, v), 48.0, 320.0);
        assert!(rms < 0.1, "rms {rms} worst {worst}");

        // A single 32 px pass cannot see a 20 px shift.
        let single = run(&a, &b, vec![32]);
        let (rms1, _, _) = errors(&single.field, |_, _| (u, v), 48.0, 320.0);
        assert!(rms1 > 1.0, "single pass unexpectedly fine: {rms1}");
    }

    #[test]
    fn solid_body_rotation() {
        let (c, omega) = (127.5, 0.03); // max ~3.4 px at the corners
        let truth = move |x: f32, y: f32| (-omega * (y - c), omega * (x - c));
        let (a, b) = synthetic::particle_pair(256, 256, 6, 0.04, 1.2, truth);
        let r = run(&a, &b, vec![64, 32]);
        let (rms, worst, _) = errors(&r.field, truth, 20.0, 256.0);
        assert!(rms < 0.15, "rms {rms} worst {worst}");
        assert_directions(&r.field, truth);
    }

    #[test]
    fn linear_shear() {
        let (c, gamma) = (127.5, 0.025);
        let truth = move |_x: f32, y: f32| (gamma * (y - c), 0.0);
        let (a, b) = synthetic::particle_pair(256, 256, 8, 0.04, 1.2, truth);
        let r = run(&a, &b, vec![64, 32]);
        let (rms, worst, _) = errors(&r.field, truth, 20.0, 256.0);
        assert!(rms < 0.15, "rms {rms} worst {worst}");
        assert_directions(&r.field, truth);
    }

    fn assert_directions(f: &VectorField, truth: impl Fn(f32, f32) -> (f32, f32)) {
        for (iy, &y) in f.ys().iter().enumerate() {
            for (ix, &x) in f.xs().iter().enumerate() {
                let (u, v) = truth(x, y);
                if (u * u + v * v).sqrt() > 0.5 {
                    let dot = u * f.dx[[iy, ix]] + v * f.dy[[iy, ix]];
                    assert!(dot > 0.0, "vector at ({x}, {y}) points the wrong way");
                }
            }
        }
    }

    #[test]
    fn blank_regions_get_no_signal() {
        let (mut a, mut b) = synthetic::particle_pair(128, 128, 9, 0.04, 1.2, |_, _| (1.0, 0.0));
        // Left half dark except faint noise.
        for img in [&mut a, &mut b] {
            img.slice_mut(ndarray::s![.., ..64]).fill(0.0);
        }
        synthetic::add_noise(&mut a, 0.002, 3);
        synthetic::add_noise(&mut b, 0.002, 4);
        let (pa, pb) = prep(&a, &b);
        let cfg = PivConfig {
            mask: MaskConfig {
                min_mean: 0.01,
                min_std: 0.01,
            },
            ..Default::default()
        };
        let r = PivAnalyzer::new(cfg).unwrap().analyze(&pa, &pb).unwrap();
        let f = &r.field;
        for (iy, _) in f.ys().iter().enumerate() {
            for (ix, &x) in f.xs().iter().enumerate() {
                let fl = f.flags[[iy, ix]];
                if x + 16.0 < 64.0 {
                    assert_eq!(fl, flags::NO_SIGNAL, "x={x}");
                    assert!(f.dx[[iy, ix]].is_nan());
                } else if x - 16.0 >= 64.0 {
                    assert_eq!(fl & flags::NO_SIGNAL, 0, "x={x}");
                }
            }
        }
    }

    #[test]
    fn rejects_bad_config() {
        assert!(
            PivAnalyzer::new(PivConfig {
                windows: vec![],
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            PivAnalyzer::new(PivConfig {
                windows: vec![33],
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            PivAnalyzer::new(PivConfig {
                overlap: 1.0,
                ..Default::default()
            })
            .is_err()
        );
        assert!(
            PivAnalyzer::new(PivConfig {
                final_iterations: 0,
                ..Default::default()
            })
            .is_err()
        );
    }
}
