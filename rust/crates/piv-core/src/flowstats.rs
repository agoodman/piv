//! Flow-wide statistics of one vector field: mean vorticity and kinetic energy.
//!
//! Inputs are velocity components on a regular grid (any consistent units; `piv` passes px/s)
//! and a mask of the vectors to use. Image axes point right and *down*; vorticity is reported
//! with the usual y-up sign, so **positive means counter-clockwise as seen on screen**.

use ndarray::{Array2, Zip};

/// Statistics of one field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowStats {
    /// Mean vorticity (1/s for px/s input), + = counter-clockwise on screen. NaN if undefined.
    pub vorticity_mean: f32,
    /// Mean |vorticity|. NaN if undefined.
    pub vorticity_abs_mean: f32,
    /// ½⟨u² + v²⟩ over the usable vectors (input units²). NaN if there are none.
    pub ke_mean: f32,
    /// ½ Σ (u² + v²) · step² over the usable vectors (input units² × px²); 0 if there are none.
    pub ke_sum: f32,
    /// Usable vectors / grid cells.
    pub coverage: f32,
    /// Usable vectors.
    pub n_usable: usize,
    /// Points where vorticity was defined.
    pub n_vorticity: usize,
}

/// One derivative along an axis at `i` of a line of `n` samples, spacing `h`. Central where
/// both neighbours are usable, one-sided where only one is (the centre must be usable).
fn derivative(i: usize, n: usize, h: f32, at: impl Fn(usize) -> Option<f32>) -> Option<f32> {
    let c = at(i)?;
    let lo = i.checked_sub(1).and_then(&at);
    let hi = (i + 1 < n).then(|| at(i + 1)).flatten();
    match (lo, hi) {
        (Some(l), Some(r)) => Some((r - l) / (2.0 * h)),
        (Some(l), None) => Some((c - l) / h),
        (None, Some(r)) => Some((r - c) / h),
        (None, None) => None,
    }
}

/// Vorticity ω = ∂u/∂y − ∂v/∂x in image coordinates (y down), which is the y-up ∂v/∂x − ∂u/∂y,
/// so positive is counter-clockwise on screen. `step` is the grid spacing in px. NaN where a
/// derivative along either axis has no usable neighbour.
pub fn vorticity(
    u: &Array2<f32>,
    v: &Array2<f32>,
    usable: &Array2<bool>,
    step: f32,
) -> Array2<f32> {
    let (ny, nx) = u.dim();
    let get = |a: &Array2<f32>, iy: usize, ix: usize| {
        let x = a[[iy, ix]];
        (usable[[iy, ix]] && x.is_finite()).then_some(x)
    };
    Array2::from_shape_fn((ny, nx), |(iy, ix)| {
        let du_dy = derivative(iy, ny, step, |j| get(u, j, ix));
        let dv_dx = derivative(ix, nx, step, |j| get(v, iy, j));
        match (du_dy, dv_dx) {
            (Some(a), Some(b)) => a - b,
            _ => f32::NAN,
        }
    })
}

/// Mean vorticity and kinetic energy of one field. See [`FlowStats`].
pub fn flow_stats(u: &Array2<f32>, v: &Array2<f32>, usable: &Array2<bool>, step: f32) -> FlowStats {
    let w = vorticity(u, v, usable, step);
    let (mut ws, mut wa, mut nw) = (0.0f64, 0.0f64, 0usize);
    for &x in w.iter().filter(|x| x.is_finite()) {
        ws += x as f64;
        wa += x.abs() as f64;
        nw += 1;
    }
    let (mut e, mut n) = (0.0f64, 0usize);
    Zip::from(u).and(v).and(usable).for_each(|&a, &b, &ok| {
        if ok && a.is_finite() && b.is_finite() {
            e += 0.5 * (a as f64 * a as f64 + b as f64 * b as f64);
            n += 1;
        }
    });
    let mean = |s: f64, n: usize| {
        if n > 0 {
            (s / n as f64) as f32
        } else {
            f32::NAN
        }
    };
    FlowStats {
        vorticity_mean: mean(ws, nw),
        vorticity_abs_mean: mean(wa, nw),
        ke_mean: mean(e, n),
        ke_sum: (e * (step as f64).powi(2)) as f32,
        coverage: n as f32 / u.len().max(1) as f32,
        n_usable: n,
        n_vorticity: nw,
    }
}

/// Estimates the random error of one vector component (same units as `u`, `v`) from how much
/// each usable vector differs from the median of its usable 8-neighbours. Real small-scale flow
/// also counts as "error", so this is an upper bound. NaN with too few points.
pub fn neighbour_noise(u: &Array2<f32>, v: &Array2<f32>, usable: &Array2<bool>) -> f32 {
    let (ny, nx) = u.dim();
    let mut resid = Vec::new();
    let mut nb = Vec::with_capacity(8);
    for a in [u, v] {
        for iy in 0..ny {
            for ix in 0..nx {
                if !usable[[iy, ix]] || !a[[iy, ix]].is_finite() {
                    continue;
                }
                nb.clear();
                for jy in iy.saturating_sub(1)..(iy + 2).min(ny) {
                    for jx in ix.saturating_sub(1)..(ix + 2).min(nx) {
                        let x = a[[jy, jx]];
                        if (jy, jx) != (iy, ix) && usable[[jy, jx]] && x.is_finite() {
                            nb.push(x);
                        }
                    }
                }
                if nb.len() < 5 {
                    continue;
                }
                let k = nb.len() / 2;
                let med = *nb.select_nth_unstable_by(k, f32::total_cmp).1;
                resid.push((a[[iy, ix]] - med).abs());
            }
        }
    }
    if resid.len() < 20 {
        return f32::NAN;
    }
    let k = resid.len() / 2;
    let mad = *resid.select_nth_unstable_by(k, f32::total_cmp).1;
    // MAD → σ for a normal residual; the neighbour median itself carries ≈ π/(2·8) of the
    // variance of one vector.
    1.4826 * mad / (1.0 + std::f32::consts::PI / 16.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 21;
    const STEP: f32 = 16.0;

    /// Velocity from a function of image position (x right, y down), all usable.
    fn field(f: impl Fn(f32, f32) -> (f32, f32)) -> (Array2<f32>, Array2<f32>, Array2<bool>) {
        let uv = Array2::from_shape_fn((N, N), |(iy, ix)| f(ix as f32 * STEP, iy as f32 * STEP));
        (
            uv.mapv(|p| p.0),
            uv.mapv(|p| p.1),
            Array2::from_elem((N, N), true),
        )
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn solid_body_rotation_counter_clockwise_on_screen_is_positive() {
        // Counter-clockwise on screen, y up: u = −Ωy′, v = Ωx′. With y′ = −y and v = −v′:
        // u = Ωy, v = −Ωx (about the centre).
        let om = 3.0;
        let c = (N - 1) as f32 * STEP / 2.0;
        let (u, v, ok) = field(|x, y| (om * (y - c), -om * (x - c)));
        let w = vorticity(&u, &v, &ok, STEP);
        assert!(w.iter().all(|&x| close(x, 2.0 * om, 1e-4)), "{w:?}");
        let s = flow_stats(&u, &v, &ok, STEP);
        assert!(close(s.vorticity_mean, 2.0 * om, 1e-4));
        assert!(close(s.vorticity_abs_mean, 2.0 * om, 1e-4));
        assert_eq!(s.n_vorticity, N * N);
    }

    #[test]
    fn uniform_translation() {
        let (u, v, ok) = field(|_, _| (3.0, -4.0));
        let s = flow_stats(&u, &v, &ok, STEP);
        assert_eq!(s.vorticity_mean, 0.0);
        assert_eq!(s.vorticity_abs_mean, 0.0);
        assert!(close(s.ke_mean, 12.5, 1e-5));
        assert!(close(s.ke_sum, 12.5 * (N * N) as f32 * STEP * STEP, 1.0));
        assert_eq!(s.coverage, 1.0);
    }

    #[test]
    fn shear() {
        // u = γ·y (y down): the bottom moves right relative to the top, like a wheel turning
        // counter-clockwise on screen, so ω = +γ.
        let g = 0.5;
        let (u, v, ok) = field(|_, y| (g * y, 0.0));
        let s = flow_stats(&u, &v, &ok, STEP);
        assert!(close(s.vorticity_mean, g, 1e-5));
    }

    #[test]
    fn opposite_vortices_cancel_in_the_signed_mean() {
        // Left half turns one way, right half the other: u = 0, v = sin(2πx/L).
        let l = N as f32 * STEP;
        let (u, v, ok) = field(|x, _| (0.0, (2.0 * std::f32::consts::PI * x / l).sin() * 10.0));
        let s = flow_stats(&u, &v, &ok, STEP);
        assert!(s.vorticity_mean.abs() < 0.1 * s.vorticity_abs_mean, "{s:?}");
        // mean |dv/dx| = 10 · 2π/L · 2/π = 40/L.
        assert!(
            close(s.vorticity_abs_mean, 40.0 / l, 0.1 * 40.0 / l),
            "{s:?}"
        );
    }

    #[test]
    fn holes_and_edges() {
        let (u, v, mut ok) = field(|x, y| (y * 0.1, x * 0.2));
        // ω = ∂u/∂y − ∂v/∂x = 0.1 − 0.2 = −0.1 everywhere, including one-sided edges.
        ok[[5, 5]] = false;
        let w = vorticity(&u, &v, &ok, STEP);
        assert!(w[[5, 5]].is_nan());
        // Neighbours of the hole fall back to one-sided differences.
        for p in [[4, 5], [6, 5], [5, 4], [5, 6], [0, 0], [N - 1, N - 1]] {
            assert!(close(w[p], -0.1, 1e-5), "{p:?}: {}", w[p]);
        }
        // An isolated vector has no derivative.
        let mut lonely = Array2::from_elem((N, N), false);
        lonely[[3, 3]] = true;
        let s = flow_stats(&u, &v, &lonely, STEP);
        assert_eq!(s.n_vorticity, 0);
        assert!(s.vorticity_mean.is_nan());
        assert_eq!(s.n_usable, 1);
        // NaN values count as unusable.
        let mut un = u.clone();
        un[[0, 0]] = f32::NAN;
        let s = flow_stats(&un, &v, &ok, STEP);
        assert_eq!(s.n_usable, N * N - 2);
    }

    #[test]
    fn empty_field() {
        let (u, v, _) = field(|_, _| (1.0, 1.0));
        let s = flow_stats(&u, &v, &Array2::from_elem((N, N), false), STEP);
        assert!(s.ke_mean.is_nan() && s.vorticity_mean.is_nan());
        assert_eq!((s.ke_sum, s.coverage, s.n_usable), (0.0, 0.0, 0));
    }

    #[test]
    fn neighbour_noise_recovers_sigma() {
        // Deterministic pseudo-random normal noise (Box–Muller on an LCG).
        let mut state = 12345u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
        };
        let mut normal = || {
            let (a, b) = (uniform().max(1e-9), uniform());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f32::consts::PI * b).cos()
        };
        let n = 80;
        let u = Array2::from_shape_fn((n, n), |_| 2.0 * normal());
        let v = Array2::from_shape_fn((n, n), |_| 2.0 * normal());
        let ok = Array2::from_elem((n, n), true);
        let s = neighbour_noise(&u, &v, &ok);
        assert!((s - 2.0).abs() < 0.15, "{s}");
    }
}
