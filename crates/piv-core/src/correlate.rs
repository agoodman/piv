//! FFT-based cross-correlation of square interrogation windows.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

use crate::Frame;

/// Result of correlating one window pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    /// Displacement of window B relative to window A, in pixels (sub-pixel).
    pub dx: f32,
    pub dy: f32,
    /// Normalized correlation coefficient at the primary peak (`≤ 1`).
    pub height: f32,
    /// Primary-to-secondary peak ratio; large means unambiguous.
    pub ppr: f32,
}

/// Upper bound reported for the peak ratio when there is no meaningful secondary peak.
pub const MAX_PPR: f32 = 1000.0;

/// FFT plans for one window size. Cheap to share between threads; each thread needs its own
/// [`Workspace`].
pub struct CorrelationPlan {
    n: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
}

/// Per-thread scratch buffers for a [`CorrelationPlan`].
pub struct Workspace {
    a: Vec<f32>,
    b: Vec<f32>,
    /// Spectra stored column-major (`m` columns of length `n`) so the column FFTs run over
    /// contiguous memory.
    ta: Vec<Complex<f32>>,
    tb: Vec<Complex<f32>>,
    row: Vec<f32>,
    row_spec: Vec<Complex<f32>>,
    corr: Vec<f32>,
    scratch: Vec<Complex<f32>>,
}

impl CorrelationPlan {
    /// `n` is the window side length; it must be even and at least 4.
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 4 && n % 2 == 0,
            "window size must be even and ≥ 4, got {n}"
        );
        let mut real = RealFftPlanner::<f32>::new();
        let mut cplx = FftPlanner::<f32>::new();
        Self {
            n,
            r2c: real.plan_fft_forward(n),
            c2r: real.plan_fft_inverse(n),
            fwd: cplx.plan_fft_forward(n),
            inv: cplx.plan_fft_inverse(n),
        }
    }

    pub fn window(&self) -> usize {
        self.n
    }

    pub fn workspace(&self) -> Workspace {
        let (n, m) = (self.n, self.n / 2 + 1);
        let scratch_len = [
            self.r2c.get_scratch_len(),
            self.c2r.get_scratch_len(),
            self.fwd.get_inplace_scratch_len(),
            self.inv.get_inplace_scratch_len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        Workspace {
            a: vec![0.0; n * n],
            b: vec![0.0; n * n],
            ta: vec![Complex::default(); m * n],
            tb: vec![Complex::default(); m * n],
            row: vec![0.0; n],
            row_spec: vec![Complex::default(); m],
            corr: vec![0.0; n * n],
            scratch: vec![Complex::default(); scratch_len],
        }
    }

    /// Correlates the `n × n` window of `a` whose top-left corner is at `(ax, ay)` with the
    /// window of `b` at `(bx, by)`. Origins may be fractional, in which case the window is
    /// sampled by bilinear interpolation; the window plus one pixel must lie inside the image.
    /// Returns `None` if either window has no intensity variation.
    pub fn correlate(
        &self,
        ws: &mut Workspace,
        a: &Frame,
        (ax, ay): (f32, f32),
        b: &Frame,
        (bx, by): (f32, f32),
    ) -> Option<Peak> {
        let n = self.n;
        let ea = load_window(&mut ws.a, a, ax, ay, n);
        let eb = load_window(&mut ws.b, b, bx, by, n);
        if ea <= 1e-12 || eb <= 1e-12 {
            return None;
        }

        self.forward(
            &ws.a,
            &mut ws.ta,
            &mut ws.row,
            &mut ws.row_spec,
            &mut ws.scratch,
        );
        self.forward(
            &ws.b,
            &mut ws.tb,
            &mut ws.row,
            &mut ws.row_spec,
            &mut ws.scratch,
        );
        ws.ta
            .iter_mut()
            .zip(&ws.tb)
            .for_each(|(fa, fb)| *fa = fa.conj() * fb);
        self.inverse(ws);

        // Unnormalized FFT round trip scales by n²; dividing by sqrt(Ea·Eb) gives a
        // correlation coefficient.
        let scale = 1.0 / ((n * n) as f64 * (ea * eb).sqrt()) as f32;
        ws.corr.iter_mut().for_each(|c| *c *= scale);
        Some(find_peak(&ws.corr, n))
    }

    fn forward(
        &self,
        window: &[f32],
        out: &mut [Complex<f32>],
        row: &mut [f32],
        row_spec: &mut [Complex<f32>],
        scratch: &mut [Complex<f32>],
    ) {
        let n = self.n;
        for (y, src) in window.chunks_exact(n).enumerate() {
            row.copy_from_slice(src);
            self.r2c
                .process_with_scratch(row, row_spec, scratch)
                .expect("buffer sizes come from the plan");
            for (c, &v) in row_spec.iter().enumerate() {
                out[c * n + y] = v;
            }
        }
        self.fwd.process_with_scratch(out, scratch);
    }

    fn inverse(&self, ws: &mut Workspace) {
        let (n, m) = (self.n, self.n / 2 + 1);
        self.inv.process_with_scratch(&mut ws.ta, &mut ws.scratch);
        for y in 0..n {
            for c in 0..m {
                ws.row_spec[c] = ws.ta[c * n + y];
            }
            // The DC and Nyquist bins of a real row are real; clear rounding residue so the
            // inverse real FFT accepts them.
            ws.row_spec[0].im = 0.0;
            ws.row_spec[m - 1].im = 0.0;
            self.c2r
                .process_with_scratch(&mut ws.row_spec, &mut ws.row, &mut ws.scratch)
                .expect("buffer sizes come from the plan");
            ws.corr[y * n..(y + 1) * n].copy_from_slice(&ws.row);
        }
    }
}

/// Copies a window into `dst` with its mean removed; returns the remaining energy Σv².
fn load_window(dst: &mut [f32], img: &Frame, x: f32, y: f32, n: usize) -> f64 {
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    if tx < 1e-4 && ty < 1e-4 {
        let view = img.slice(ndarray::s![y0..y0 + n, x0..x0 + n]);
        dst.iter_mut().zip(view.iter()).for_each(|(d, &v)| *d = v);
    } else {
        let view = img.slice(ndarray::s![y0..y0 + n + 1, x0..x0 + n + 1]);
        let (w00, w01, w10, w11) = (
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        );
        for (r, out) in dst.chunks_exact_mut(n).enumerate() {
            let (top, bot) = (view.row(r), view.row(r + 1));
            for (c, d) in out.iter_mut().enumerate() {
                *d = w00 * top[c] + w01 * top[c + 1] + w10 * bot[c] + w11 * bot[c + 1];
            }
        }
    }
    let mean = dst.iter().map(|&v| v as f64).sum::<f64>() / dst.len() as f64;
    let mut energy = 0.0;
    for d in dst.iter_mut() {
        *d -= mean as f32;
        energy += (*d as f64) * (*d as f64);
    }
    energy
}

/// Locates the highest peak of a circular `n × n` correlation plane, refines it with a
/// three-point Gaussian fit (parabolic when values are not all positive), and measures the
/// ratio to the highest other local maximum.
pub fn find_peak(corr: &[f32], n: usize) -> Peak {
    let at = |y: usize, x: usize| corr[(y % n) * n + (x % n)];
    let is_local_max = |y: usize, x: usize| {
        let v = at(y, x);
        [
            (n - 1, n - 1),
            (n - 1, 0),
            (n - 1, 1),
            (0, n - 1),
            (0, 1),
            (1, n - 1),
            (1, 0),
            (1, 1),
        ]
        .iter()
        .all(|&(dy, dx)| at(y + dy, x + dx) <= v)
    };

    let (mut best, mut by, mut bx) = (f32::NEG_INFINITY, 0, 0);
    for (i, &v) in corr.iter().enumerate() {
        if v > best {
            (best, by, bx) = (v, i / n, i % n);
        }
    }
    let mut second = f32::NEG_INFINITY;
    for y in 0..n {
        for x in 0..n {
            let v = at(y, x);
            if (y, x) != (by, bx) && v > second && is_local_max(y, x) {
                second = v;
            }
        }
    }

    // The overlap between two n-wide windows shrinks linearly with displacement, which biases
    // the peak toward zero (Westerweel 1997). Undo that weighting on the fit points.
    let unbias = |k: usize| 1.0 / (1.0 - k.min(n - k) as f32 / n as f32);
    let at_unbiased = |y: usize, x: usize| at(y, x) * unbias(y % n) * unbias(x % n);

    let sub = |m: f32, c: f32, p: f32| -> f32 {
        let d = if m > 0.0 && c > 0.0 && p > 0.0 {
            let (lm, lc, lp) = (m.ln(), c.ln(), p.ln());
            (lm - lp) / (2.0 * (lm - 2.0 * lc + lp))
        } else {
            (m - p) / (2.0 * (m - 2.0 * c + p))
        };
        if d.is_finite() {
            d.clamp(-0.5, 0.5)
        } else {
            0.0
        }
    };
    // After unbiasing, a neighbour can overtake the raw maximum; climb to the true peak.
    let (mut by, mut bx) = (by, bx);
    for _ in 0..3 {
        let mut next = (by, bx);
        for (dy, dx) in [(n - 1, 0), (1, 0), (0, n - 1), (0, 1)] {
            let (y, x) = ((by + dy) % n, (bx + dx) % n);
            if at_unbiased(y, x) > at_unbiased(next.0, next.1) {
                next = (y, x);
            }
        }
        if next == (by, bx) {
            break;
        }
        (by, bx) = next;
    }
    let centre = at_unbiased(by, bx);
    let sx = sub(at_unbiased(by, bx + n - 1), centre, at_unbiased(by, bx + 1));
    let sy = sub(at_unbiased(by + n - 1, bx), centre, at_unbiased(by + 1, bx));

    let wrap = |k: usize| {
        if k >= n / 2 {
            k as f32 - n as f32
        } else {
            k as f32
        }
    };
    let ppr = if second > 0.0 {
        (best / second).min(MAX_PPR)
    } else {
        MAX_PPR
    };
    Peak {
        dx: wrap(bx) + sx,
        dy: wrap(by) + sy,
        height: best,
        ppr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{self, Rng};

    #[test]
    fn integer_shift_is_exact_for_circular_content() {
        // A periodic random pattern shifted circularly correlates perfectly.
        let n = 32;
        let mut rng = Rng::new(7);
        let a = Frame::from_shape_fn((n, n), |_| rng.uniform());
        let (sx, sy) = (5usize, n - 3); // +5 in x, −3 in y
        let b = Frame::from_shape_fn((n, n), |(y, x)| a[[(y + n - sy) % n, (x + n - sx) % n]]);
        let plan = CorrelationPlan::new(n);
        let mut ws = plan.workspace();
        let p = plan
            .correlate(&mut ws, &a, (0.0, 0.0), &b, (0.0, 0.0))
            .unwrap();
        // Bias correction assumes non-periodic content, so allow a small error here.
        assert!(
            (p.dx - 5.0).abs() < 0.02 && (p.dy + 3.0).abs() < 0.02,
            "{p:?}"
        );
        assert!((p.height - 1.0).abs() < 1e-4, "{p:?}");
        assert!(p.ppr > 2.0);
    }

    #[test]
    fn subpixel_shift_of_particles() {
        let (a, b) = synthetic::particle_pair(64, 64, 3, 0.05, 1.2, |_, _| (1.3, -0.6));
        let plan = CorrelationPlan::new(32);
        let mut ws = plan.workspace();
        let p = plan
            .correlate(&mut ws, &a, (16.0, 16.0), &b, (16.0, 16.0))
            .unwrap();
        assert!(
            (p.dx - 1.3).abs() < 0.1 && (p.dy + 0.6).abs() < 0.1,
            "{p:?}"
        );
    }

    #[test]
    fn fractional_origin_interpolates() {
        let a = Frame::from_shape_fn((40, 40), |(y, x)| (x + 2 * y) as f32);
        let mut dst = vec![0.0; 16 * 16];
        load_window(&mut dst, &a, 3.5, 2.25, 16);
        // A linear ramp is reproduced exactly; after mean removal compare differences.
        assert!((dst[1] - dst[0] - 1.0).abs() < 1e-4 && (dst[16] - dst[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn flat_window_has_no_peak() {
        let a = Frame::from_elem((32, 32), 0.3);
        let plan = CorrelationPlan::new(32);
        let mut ws = plan.workspace();
        assert!(
            plan.correlate(&mut ws, &a, (0.0, 0.0), &a, (0.0, 0.0))
                .is_none()
        );
    }

    #[test]
    fn peak_ratio_uses_local_maxima() {
        // A broad single peak: neighbours of the maximum must not count as a second peak.
        let n = 16;
        let mut c = vec![0.0f32; n * n];
        for y in 0..n {
            for x in 0..n {
                let (dy, dx) = (y.min(n - y) as f32, x.min(n - x) as f32);
                c[y * n + x] = (-(dx * dx + dy * dy) / 8.0).exp();
            }
        }
        c[8 * n + 8] = 0.25; // a distinct secondary peak
        let p = find_peak(&c, n);
        assert_eq!((p.dx, p.dy), (0.0, 0.0));
        assert!((p.ppr - 4.0).abs() < 1e-3, "{p:?}");
    }
}
