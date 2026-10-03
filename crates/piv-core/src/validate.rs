//! Outlier detection and replacement.

use serde::{Deserialize, Serialize};

use crate::field::{VectorField, flags};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Vectors whose primary/secondary peak ratio is below this are flagged `LOW_PPR`.
    pub ppr_threshold: f32,
    /// Normalized median test threshold (Westerweel & Scarano 2005 recommend 2).
    pub median_threshold: f32,
    /// Normalized median test noise level ε, in pixels. 0.1 is the textbook value for particle
    /// images; 0.3 rejected fewer good vectors on dry-ice fog without admitting worse ones.
    pub median_eps: f32,
    /// Replace rejected vectors with the median of valid neighbours.
    pub replace: bool,
}

/// Limits that depend on the interrogation setup rather than on the user's tolerances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Largest displacement magnitude (px) the analysis can measure; larger is flagged
    /// `OUT_OF_RANGE`. Half the first-pass window: beyond that the correlation peak aliases.
    pub max_displacement: f32,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            ppr_threshold: 1.3,
            median_threshold: 2.0,
            median_eps: 0.1,
            replace: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ValidationSummary {
    pub no_signal: usize,
    pub low_ppr: usize,
    pub median_outliers: usize,
    pub out_of_range: usize,
    pub replaced: usize,
}

/// Flags low-confidence vectors and outliers in place and (optionally) replaces them.
///
/// `NO_SIGNAL` vectors are left as NaN: they are missing data, not errors.
pub fn validate(
    field: &mut VectorField,
    cfg: &ValidationConfig,
    limits: Limits,
) -> ValidationSummary {
    let (ny, nx) = field.dx.dim();
    let mut summary = ValidationSummary::default();

    for (((f, &ppr), &u), &v) in field
        .flags
        .iter_mut()
        .zip(&field.peak_ratio)
        .zip(&field.dx)
        .zip(&field.dy)
    {
        if *f & flags::NO_SIGNAL != 0 {
            summary.no_signal += 1;
            continue;
        }
        if u.hypot(v) > limits.max_displacement {
            *f |= flags::OUT_OF_RANGE;
            summary.out_of_range += 1;
        }
        if ppr < cfg.ppr_threshold {
            *f |= flags::LOW_PPR;
            summary.low_ppr += 1;
        }
    }

    let usable = |f: u8| f & flags::INVALID == 0;
    let snapshot = field.flags.clone();
    let mut nu = Vec::with_capacity(8);
    let mut nv = Vec::with_capacity(8);
    for iy in 0..ny {
        for ix in 0..nx {
            if !usable(snapshot[[iy, ix]]) {
                continue;
            }
            collect_neighbours(field, &snapshot, usable, ix, iy, 1, &mut nu, &mut nv);
            if nu.len() < 2 {
                continue;
            }
            let ru = residual(field.dx[[iy, ix]], &mut nu, cfg.median_eps);
            let rv = residual(field.dy[[iy, ix]], &mut nv, cfg.median_eps);
            if (ru * ru + rv * rv).sqrt() > cfg.median_threshold {
                field.flags[[iy, ix]] |= flags::MEDIAN_OUTLIER;
                summary.median_outliers += 1;
            }
        }
    }

    if cfg.replace {
        let snapshot = field.flags.clone();
        let rejected = |f: u8| f & flags::NO_SIGNAL == 0 && f & flags::INVALID != 0;
        for iy in 0..ny {
            for ix in 0..nx {
                if !rejected(snapshot[[iy, ix]]) {
                    continue;
                }
                let mut replaced = false;
                for radius in 1..=2 {
                    collect_neighbours(field, &snapshot, usable, ix, iy, radius, &mut nu, &mut nv);
                    if !nu.is_empty() {
                        field.dx[[iy, ix]] = median(&mut nu);
                        field.dy[[iy, ix]] = median(&mut nv);
                        field.flags[[iy, ix]] |= flags::REPLACED;
                        summary.replaced += 1;
                        replaced = true;
                        break;
                    }
                }
                if !replaced {
                    field.dx[[iy, ix]] = f32::NAN;
                    field.dy[[iy, ix]] = f32::NAN;
                }
            }
        }
    }
    summary
}

#[allow(clippy::too_many_arguments)]
fn collect_neighbours(
    field: &VectorField,
    flags: &ndarray::Array2<u8>,
    usable: impl Fn(u8) -> bool,
    ix: usize,
    iy: usize,
    radius: usize,
    nu: &mut Vec<f32>,
    nv: &mut Vec<f32>,
) {
    let (ny, nx) = field.dx.dim();
    nu.clear();
    nv.clear();
    for y in iy.saturating_sub(radius)..=(iy + radius).min(ny - 1) {
        for x in ix.saturating_sub(radius)..=(ix + radius).min(nx - 1) {
            if (x, y) != (ix, iy) && usable(flags[[y, x]]) {
                nu.push(field.dx[[y, x]]);
                nv.push(field.dy[[y, x]]);
            }
        }
    }
}

/// Normalized median residual of `value` against its neighbours (which get reordered).
fn residual(value: f32, neighbours: &mut [f32], eps: f32) -> f32 {
    let med = median(neighbours);
    let mut dev: Vec<f32> = neighbours.iter().map(|v| (v - med).abs()).collect();
    (value - med).abs() / (median(&mut dev) + eps)
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;

    fn uniform_field(u: f32, v: f32) -> VectorField {
        let mut f = VectorField::new(Grid::new(160, 160, 32, 0.5).unwrap());
        f.dx.fill(u);
        f.dy.fill(v);
        f.peak_ratio.fill(5.0);
        f.flags.fill(0);
        f
    }

    #[test]
    fn catches_and_replaces_injected_outliers() {
        let mut f = uniform_field(2.0, -1.0);
        // Small realistic jitter so the median deviation isn't zero.
        for (i, v) in f.dx.iter_mut().enumerate() {
            *v += 0.05 * ((i * 7 % 5) as f32 - 2.0);
        }
        let outliers = [(2, 3), (5, 5), (8, 0)];
        for &(iy, ix) in &outliers {
            f.dx[[iy, ix]] = 9.0;
            f.dy[[iy, ix]] = 4.0;
        }
        let s = validate(&mut f, &ValidationConfig::default(), NO_LIMIT);
        assert_eq!(s.median_outliers, outliers.len(), "{s:?}");
        assert_eq!(s.replaced, outliers.len());
        for &(iy, ix) in &outliers {
            let fl = f.flags[[iy, ix]];
            assert!(fl & flags::MEDIAN_OUTLIER != 0 && fl & flags::REPLACED != 0);
            assert!((f.dx[[iy, ix]] - 2.0).abs() < 0.2 && (f.dy[[iy, ix]] + 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn low_peak_ratio_is_flagged_and_no_signal_untouched() {
        let mut f = uniform_field(1.0, 1.0);
        f.peak_ratio[[4, 4]] = 1.05;
        f.flags[[0, 0]] = flags::NO_SIGNAL;
        f.dx[[0, 0]] = f32::NAN;
        f.dy[[0, 0]] = f32::NAN;
        let s = validate(&mut f, &ValidationConfig::default(), NO_LIMIT);
        assert_eq!((s.low_ppr, s.no_signal, s.median_outliers), (1, 1, 0));
        assert_eq!(f.flags[[4, 4]], flags::LOW_PPR | flags::REPLACED);
        assert_eq!(f.flags[[0, 0]], flags::NO_SIGNAL);
        assert!(f.dx[[0, 0]].is_nan());
    }

    const NO_LIMIT: Limits = Limits {
        max_displacement: f32::INFINITY,
    };

    #[test]
    fn out_of_range_vectors_are_flagged_and_replaced() {
        let mut f = uniform_field(1.0, 0.5);
        f.dx[[3, 3]] = 40.0;
        let s = validate(
            &mut f,
            &ValidationConfig::default(),
            Limits {
                max_displacement: 32.0,
            },
        );
        assert_eq!(s.out_of_range, 1);
        assert_eq!(f.flags[[3, 3]] & flags::OUT_OF_RANGE, flags::OUT_OF_RANGE);
        assert_eq!(f.flags[[3, 3]] & flags::REPLACED, flags::REPLACED);
        assert_eq!(f.dx[[3, 3]], 1.0);
    }

    #[test]
    fn median_handles_even_and_odd() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
