//! Post-processing filters for a time series of vector fields.
//!
//! Fields are stacked as `[pair, ny, nx]`. Every filter is NaN-aware: only vectors marked
//! usable contribute, and an output is produced only where the `gate` allows it (normally
//! "this window had seeding in this pair"), so smoothing never paints velocity into empty
//! regions.
//!
//! Each filter can predict every point while leaving out the samples that share data with it
//! (`leave_out`): the centre itself, pairs that share a frame with it, and windows that overlap
//! it. Comparing that prediction with the raw vector ([`cross_validation_errors`]) gives a fair
//! comparison of filter settings: too little smoothing leaves noise in the prediction, too much
//! adds bias. Leaving out only the centre would be optimistic, because overlapping windows and
//! shared frames make neighbouring errors correlated.

use ndarray::{Array3, Axis, Zip};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TemporalKernel {
    Mean,
    Median,
}

/// A smoothed series: displacement plus how many raw vectors supported each output.
#[derive(Debug, Clone)]
pub struct Smoothed {
    pub dx: Array3<f32>,
    pub dy: Array3<f32>,
    /// Number of usable raw vectors in the window (may be below `min_support` where there is no
    /// output).
    pub support: Array3<u16>,
}

/// Moving mean or median over `window` consecutive pairs (odd, centred) at each grid point.
///
/// `usable[t, y, x]` marks raw vectors that may contribute; `gate` marks where an output is
/// wanted. Outputs with fewer than `min_support` contributing vectors are NaN. `leave_out`
/// lists time offsets (e.g. `[0, -5, 5]`) to skip, for cross-validation. Each skipped non-zero
/// offset is replaced by the next one further out, so a leave-out prediction draws on the same
/// number of neighbouring pairs as the filter itself (`window - 1` once the centre is left out).
#[allow(clippy::too_many_arguments)]
pub fn temporal(
    dx: &Array3<f32>,
    dy: &Array3<f32>,
    usable: &Array3<bool>,
    gate: &Array3<bool>,
    kernel: TemporalKernel,
    window: usize,
    min_support: usize,
    leave_out: &[isize],
) -> Smoothed {
    assert!(window % 2 == 1, "temporal window must be odd");
    let (nt, ny, nx) = dx.dim();
    let half = window / 2;
    let mut offsets: Vec<isize> = Vec::with_capacity(window);
    if !leave_out.contains(&0) {
        offsets.push(0);
    }
    for sign in [-1, 1] {
        let mut k = 1;
        let mut taken = 0;
        while taken < half {
            if !leave_out.contains(&(sign * k)) {
                offsets.push(sign * k);
                taken += 1;
            }
            k += 1;
        }
    }
    let mut out = Smoothed {
        dx: Array3::from_elem((nt, ny, nx), f32::NAN),
        dy: Array3::from_elem((nt, ny, nx), f32::NAN),
        support: Array3::zeros((nt, ny, nx)),
    };

    // Each grid point is an independent time series; process them in parallel.
    let columns: Vec<(Vec<f32>, Vec<f32>, Vec<u16>)> = (0..ny * nx)
        .into_par_iter()
        .map(|k| {
            let (y, x) = (k / nx, k % nx);
            let mut ou = vec![f32::NAN; nt];
            let mut ov = vec![f32::NAN; nt];
            let mut os = vec![0u16; nt];
            let (mut bu, mut bv) = (Vec::with_capacity(window), Vec::with_capacity(window));
            for t in 0..nt {
                if !gate[[t, y, x]] {
                    continue;
                }
                bu.clear();
                bv.clear();
                for &o in &offsets {
                    let s = t as isize + o;
                    if s < 0 || s >= nt as isize || !usable[[s as usize, y, x]] {
                        continue;
                    }
                    let s = s as usize;
                    bu.push(dx[[s, y, x]]);
                    bv.push(dy[[s, y, x]]);
                }
                os[t] = bu.len() as u16;
                if bu.len() >= min_support.max(1) {
                    (ou[t], ov[t]) = match kernel {
                        TemporalKernel::Mean => (mean(&bu), mean(&bv)),
                        TemporalKernel::Median => (median(&mut bu), median(&mut bv)),
                    };
                }
            }
            (ou, ov, os)
        })
        .collect();

    for (k, (ou, ov, os)) in columns.into_iter().enumerate() {
        let (y, x) = (k / nx, k % nx);
        for t in 0..nt {
            out.dx[[t, y, x]] = ou[t];
            out.dy[[t, y, x]] = ov[t];
            out.support[[t, y, x]] = os[t];
        }
    }
    out
}

/// Gaussian low-pass of each pair's field, `sigma` in grid cells, by normalized convolution:
/// missing (NaN) neighbours are skipped and the remaining weights renormalized. An output needs
/// at least `min_weight` of the kernel's (non-left-out) weight to be present. `support` counts
/// the finite neighbours that contributed.
///
/// With `leave_out = Some(r)`, every point within Chebyshev distance `r` of the centre is
/// skipped. When that removes more than half of the kernel's weight, what is left would be a
/// different estimator (a ring interpolator) rather than this filter, so nothing is predicted
/// and every output is NaN.
pub fn spatial(
    dx: &Array3<f32>,
    dy: &Array3<f32>,
    gate: &Array3<bool>,
    sigma: f32,
    min_weight: f32,
    leave_out: Option<usize>,
) -> Smoothed {
    let excluded =
        |i: isize, j: isize| leave_out.is_some_and(|r| i.unsigned_abs().max(j.unsigned_abs()) <= r);
    let r = ((2.5 * sigma).ceil() as isize).max(1);
    let kernel: Vec<Vec<f32>> = (-r..=r)
        .map(|j| {
            (-r..=r)
                .map(|i| (-((i * i + j * j) as f32) / (2.0 * sigma * sigma)).exp())
                .collect()
        })
        .collect();
    let weight = |keep: &dyn Fn(isize, isize) -> bool| -> f32 {
        (-r..=r)
            .flat_map(|j| (-r..=r).map(move |i| (i, j)))
            .filter(|&(i, j)| keep(i, j))
            .map(|(i, j)| kernel[(j + r) as usize][(i + r) as usize])
            .sum()
    };
    let full = weight(&|i, j| !excluded(i, j));
    let mut out = Smoothed {
        dx: Array3::from_elem(dx.dim(), f32::NAN),
        dy: Array3::from_elem(dx.dim(), f32::NAN),
        support: Array3::zeros(dx.dim()),
    };
    if full < 0.5 * weight(&|_, _| true) {
        return out;
    }

    Zip::from(out.dx.outer_iter_mut())
        .and(out.dy.outer_iter_mut())
        .and(out.support.outer_iter_mut())
        .and(dx.outer_iter())
        .and(dy.outer_iter())
        .and(gate.outer_iter())
        .par_for_each(|mut ox, mut oy, mut os, u, v, g| {
            let (ny, nx) = u.dim();
            for y in 0..ny {
                for x in 0..nx {
                    if !g[[y, x]] {
                        continue;
                    }
                    let (mut su, mut sv, mut sw, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u16);
                    for j in -r..=r {
                        let yy = y as isize + j;
                        if yy < 0 || yy >= ny as isize {
                            continue;
                        }
                        for i in -r..=r {
                            let xx = x as isize + i;
                            if xx < 0 || xx >= nx as isize || excluded(i, j) {
                                continue;
                            }
                            let (a, b) =
                                (u[[yy as usize, xx as usize]], v[[yy as usize, xx as usize]]);
                            if a.is_finite() && b.is_finite() {
                                let w = kernel[(j + r) as usize][(i + r) as usize];
                                su += w * a;
                                sv += w * b;
                                sw += w;
                                n += 1;
                            }
                        }
                    }
                    os[[y, x]] = n;
                    if sw >= min_weight * full && sw > 0.0 {
                        ox[[y, x]] = su / sw;
                        oy[[y, x]] = sv / sw;
                    }
                }
            }
        });
    out
}

/// Distance (px) between each usable raw vector and its leave-out prediction, in iteration
/// order over the usable vectors; NaN where nothing was predicted. Compare settings with
/// [`median_error`] over vectors that all of them predict, or the easier points win.
pub fn cross_validation_errors(
    raw_dx: &Array3<f32>,
    raw_dy: &Array3<f32>,
    usable: &Array3<bool>,
    pred_dx: &Array3<f32>,
    pred_dy: &Array3<f32>,
) -> Vec<f32> {
    let mut errs: Vec<f32> = Vec::new();
    Zip::from(raw_dx)
        .and(raw_dy)
        .and(usable)
        .and(pred_dx)
        .and(pred_dy)
        .for_each(|&u, &v, &ok, &pu, &pv| {
            if ok {
                errs.push((u - pu).hypot(v - pv));
            }
        });
    errs
}

/// Median of the finite errors where `keep` is true (all of them without a mask), and how many
/// that was.
pub fn median_error(errors: &[f32], keep: Option<&[bool]>) -> (f32, usize) {
    let mut e: Vec<f32> = errors
        .iter()
        .enumerate()
        .filter(|&(i, e)| e.is_finite() && keep.is_none_or(|k| k[i]))
        .map(|(_, &e)| e)
        .collect();
    let n = e.len();
    if n == 0 {
        return (f32::NAN, 0);
    }
    (median(&mut e), n)
}

/// Median of |d(t+1) − d(t)| over grid points defined at both times (px).
pub fn pair_to_pair_change(dx: &Array3<f32>, dy: &Array3<f32>) -> f32 {
    let nt = dx.len_of(Axis(0));
    let mut d = Vec::new();
    for t in 1..nt {
        Zip::from(dx.index_axis(Axis(0), t))
            .and(dy.index_axis(Axis(0), t))
            .and(dx.index_axis(Axis(0), t - 1))
            .and(dy.index_axis(Axis(0), t - 1))
            .for_each(|&a, &b, &c, &e| {
                let m = (a - c).hypot(b - e);
                if m.is_finite() {
                    d.push(m);
                }
            });
    }
    if d.is_empty() {
        f32::NAN
    } else {
        median(&mut d)
    }
}

/// Median of |d − mean of its 4 neighbours| over points where all five are defined (px).
pub fn spatial_roughness(dx: &Array3<f32>, dy: &Array3<f32>) -> f32 {
    let mut d = Vec::new();
    for (u, v) in dx.outer_iter().zip(dy.outer_iter()) {
        let (ny, nx) = u.dim();
        for y in 1..ny.saturating_sub(1) {
            for x in 1..nx.saturating_sub(1) {
                let pts = [(y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)];
                if pts.iter().any(|&p| !u[p].is_finite() || !v[p].is_finite()) {
                    continue;
                }
                let mu = (u[pts[1]] + u[pts[2]] + u[pts[3]] + u[pts[4]]) / 4.0;
                let mv = (v[pts[1]] + v[pts[2]] + v[pts[3]] + v[pts[4]]) / 4.0;
                d.push((u[pts[0]] - mu).hypot(v[pts[0]] - mv));
            }
        }
    }
    if d.is_empty() {
        f32::NAN
    } else {
        median(&mut d)
    }
}

/// Replaces values where `usable` is false with NaN.
pub fn mask(values: &Array3<f32>, usable: &Array3<bool>) -> Array3<f32> {
    let mut out = values.clone();
    Zip::from(&mut out).and(usable).for_each(|v, &ok| {
        if !ok {
            *v = f32::NAN;
        }
    });
    out
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_unstable_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::Rng;

    /// (noisy dx, noisy dy, true dx, true dy, usable)
    type Series = (
        Array3<f32>,
        Array3<f32>,
        Array3<f32>,
        Array3<f32>,
        Array3<bool>,
    );

    /// A slowly varying smooth flow plus Gaussian noise.
    fn series(nt: usize, noise: f32) -> Series {
        let (ny, nx) = (12, 16);
        let mut rng = Rng::new(1);
        let truth_u = Array3::from_shape_fn((nt, ny, nx), |(t, y, _)| {
            2.0 + 0.01 * t as f32 + 0.05 * y as f32
        });
        let truth_v = Array3::from_shape_fn((nt, ny, nx), |(t, _, x)| {
            -1.0 + 0.02 * x as f32 - 0.005 * t as f32
        });
        let du = truth_u.mapv(|v| v + noise * rng.normal());
        let dv = truth_v.mapv(|v| v + noise * rng.normal());
        let usable = Array3::from_elem((nt, ny, nx), true);
        (du, dv, truth_u, truth_v, usable)
    }

    fn rms(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
        let (s, n) = a
            .iter()
            .zip(b)
            .filter(|(x, _)| x.is_finite())
            .fold((0.0, 0), |(s, n), (x, y)| (s + (x - y).powi(2), n + 1));
        (s / n as f32).sqrt()
    }

    #[test]
    fn temporal_mean_reduces_noise_by_sqrt_window() {
        let (du, dv, tu, _, usable) = series(60, 0.5);
        let s = temporal(&du, &dv, &usable, &usable, TemporalKernel::Mean, 9, 1, &[]);
        let before = rms(&du, &tu);
        let after = rms(&s.dx, &tu);
        assert!(after < before / 2.5, "before {before} after {after}");
        assert_eq!(s.support[[30, 5, 5]], 9);
        assert_eq!(s.support[[0, 5, 5]], 5, "window is truncated at the ends");
    }

    #[test]
    fn temporal_median_rejects_outliers_and_respects_usable() {
        let (mut du, dv, tu, _, mut usable) = series(30, 0.0);
        du[[10, 3, 3]] = 50.0; // an accepted outlier: the median ignores it
        du[[11, 3, 3]] = -50.0;
        usable[[12, 3, 3]] = false; // a rejected vector: never used
        du[[12, 3, 3]] = 1e6;
        let s = temporal(
            &du,
            &dv,
            &usable,
            &usable.mapv(|_| true),
            TemporalKernel::Median,
            7,
            1,
            &[],
        );
        assert!((s.dx[[11, 3, 3]] - tu[[11, 3, 3]]).abs() < 0.05);
        assert!(
            (s.dx[[12, 3, 3]] - tu[[12, 3, 3]]).abs() < 0.05,
            "gated-in output where raw was rejected"
        );
        assert_eq!(s.support[[12, 3, 3]], 6);
    }

    #[test]
    fn gate_and_min_support_control_output() {
        let (du, dv, _, _, usable) = series(20, 0.1);
        let mut gate = usable.clone();
        gate[[5, 0, 0]] = false;
        let s = temporal(&du, &dv, &usable, &gate, TemporalKernel::Mean, 5, 4, &[]);
        assert!(s.dx[[5, 0, 0]].is_nan(), "no output where gated off");
        assert!(
            s.dx[[0, 1, 1]].is_nan(),
            "only 3 samples at t=0 < min_support 4"
        );
        assert!(s.dx[[1, 1, 1]].is_finite());
    }

    #[test]
    fn spatial_smooths_noise_preserves_linear_fields_and_skips_nan() {
        let (mut du, dv, tu, tv, usable) = series(3, 0.0);
        // Linear fields pass through an interior symmetric kernel unchanged.
        let o = spatial(&du, &dv, &usable, 1.0, 0.25, None);
        let (ou, ov) = (o.dx, o.dy);
        assert!(
            (ou[[1, 6, 8]] - tu[[1, 6, 8]]).abs() < 1e-4
                && (ov[[1, 6, 8]] - tv[[1, 6, 8]]).abs() < 1e-4
        );
        // NaN neighbours are skipped, not propagated.
        du[[1, 6, 9]] = f32::NAN;
        let o = spatial(&du, &dv, &usable, 1.0, 0.25, None);
        assert!(o.dx[[1, 6, 8]].is_finite());
        assert_eq!(
            o.support[[1, 6, 8]],
            48,
            "7×7 kernel less one missing neighbour"
        );

        let (nu, nv, tu, _, usable) = series(3, 0.5);
        let o = spatial(&nu, &nv, &usable, 1.5, 0.25, None);
        assert!(rms(&o.dx, &tu) < rms(&nu, &tu) / 2.0);
    }

    #[test]
    fn cross_validation_prefers_moderate_smoothing() {
        let (du, dv, _, _, usable) = series(80, 0.5);
        let cv = |w: usize| {
            let s = temporal(&du, &dv, &usable, &usable, TemporalKernel::Mean, w, 1, &[0]);
            median_error(
                &cross_validation_errors(&du, &dv, &usable, &s.dx, &s.dy),
                None,
            )
            .0
        };
        // Noise dominates here, so a wider window predicts better.
        assert!(cv(9) < cv(3), "{} vs {}", cv(9), cv(3));
    }

    #[test]
    fn leave_out_skips_the_requested_samples() {
        let (mut du, dv, _, _, usable) = series(20, 0.0);
        du[[10, 4, 4]] = 100.0;
        du[[15, 4, 4]] = 100.0;
        let s = temporal(
            &du,
            &dv,
            &usable,
            &usable,
            TemporalKernel::Mean,
            11,
            1,
            &[0, 5, -5],
        );
        assert!(
            s.dx[[10, 4, 4]] < 3.0,
            "centre and ±5 left out: {}",
            s.dx[[10, 4, 4]]
        );
        assert_eq!(s.support[[10, 4, 4]], 10, "±5 replaced by ±6");

        // A 3-pair window with ±1 sharing a frame still predicts, from ±2.
        let s = temporal(
            &du,
            &dv,
            &usable,
            &usable,
            TemporalKernel::Mean,
            3,
            1,
            &[0, 1, -1],
        );
        assert_eq!(s.support[[10, 4, 4]], 2);
        assert!(s.dx[[10, 4, 4]].is_finite());

        let (mut du, dv, _, _, usable) = series(1, 0.0);
        du[[0, 6, 7]] = 100.0; // a neighbour at distance 1 of (6, 8)
        let o = spatial(&du, &dv, &usable, 1.5, 0.0, Some(1));
        assert!(
            o.dx[[0, 6, 8]] < 3.0,
            "3×3 block left out: {}",
            o.dx[[0, 6, 8]]
        );
        // A narrow kernel has most of its weight inside the left-out block: no prediction.
        let o = spatial(&du, &dv, &usable, 0.5, 0.0, Some(1));
        assert!(o.dx.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn median_error_respects_mask() {
        let e = [1.0, f32::NAN, 3.0, 10.0];
        assert_eq!(median_error(&e, None), (3.0, 3));
        assert_eq!(median_error(&e, Some(&[true, true, true, false])), (2.0, 2));
    }

    #[test]
    fn roughness_metrics() {
        let (du, dv, tu, tv, _) = series(10, 0.3);
        assert!(pair_to_pair_change(&du, &dv) > 0.3);
        assert!(pair_to_pair_change(&tu, &tv) < 0.05);
        assert!(spatial_roughness(&du, &dv) > 0.2);
        assert!(spatial_roughness(&tu, &tv) < 1e-4);
    }
}
