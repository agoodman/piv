//! Synthetic image pairs with a known displacement field, for tests and benchmarks.
//!
//! Images are rendered analytically as sums of Gaussian blobs, so a blob moved by a
//! sub-pixel amount is rendered exactly rather than interpolated.

use crate::Frame;

/// Small deterministic PRNG (SplitMix64) so tests don't depend on an external crate.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// A Gaussian blob: centre, sigma, peak amplitude.
#[derive(Debug, Clone, Copy)]
pub struct Blob {
    pub x: f32,
    pub y: f32,
    pub sigma: f32,
    pub amp: f32,
}

/// Renders blobs into an `h × w` frame (additively).
pub fn render(h: usize, w: usize, blobs: &[Blob]) -> Frame {
    let mut f = Frame::zeros((h, w));
    for b in blobs {
        let r = (3.5 * b.sigma).ceil() as isize;
        let (cx, cy) = (b.x.round() as isize, b.y.round() as isize);
        let k = 1.0 / (2.0 * b.sigma * b.sigma);
        for y in (cy - r).max(0)..=(cy + r).min(h as isize - 1) {
            for x in (cx - r).max(0)..=(cx + r).min(w as isize - 1) {
                let (dx, dy) = (x as f32 - b.x, y as f32 - b.y);
                f[[y as usize, x as usize]] += b.amp * (-(dx * dx + dy * dy) * k).exp();
            }
        }
    }
    f
}

/// Moves every blob by `field(x, y)` evaluated at the blob's midpoint position (second-order
/// accurate for smooth fields).
pub fn displace(blobs: &[Blob], field: impl Fn(f32, f32) -> (f32, f32)) -> Vec<Blob> {
    blobs
        .iter()
        .map(|b| {
            // Fixed-point iteration for the midpoint: p_b = p_a + f((p_a + p_b) / 2).
            let (mut dx, mut dy) = field(b.x, b.y);
            for _ in 0..3 {
                (dx, dy) = field(b.x + dx / 2.0, b.y + dy / 2.0);
            }
            Blob {
                x: b.x + dx,
                y: b.y + dy,
                ..*b
            }
        })
        .collect()
}

/// Random particles: `density` particles per pixel, image sigma `sigma` px.
pub fn particles(h: usize, w: usize, seed: u64, density: f32, sigma: f32) -> Vec<Blob> {
    let mut rng = Rng::new(seed);
    let margin = 20.0;
    let area = (h as f32 + 2.0 * margin) * (w as f32 + 2.0 * margin);
    (0..(density * area) as usize)
        .map(|_| Blob {
            x: rng.uniform() * (w as f32 + 2.0 * margin) - margin,
            y: rng.uniform() * (h as f32 + 2.0 * margin) - margin,
            sigma: sigma * (0.8 + 0.4 * rng.uniform()),
            amp: 0.3 + 0.7 * rng.uniform(),
        })
        .collect()
}

/// Smooth, fog-like texture: overlapping blobs of 2–6 px sigma, no resolved particles.
pub fn fog(h: usize, w: usize, seed: u64) -> Vec<Blob> {
    let mut rng = Rng::new(seed);
    let margin = 30.0;
    let area = (h as f32 + 2.0 * margin) * (w as f32 + 2.0 * margin);
    (0..(area / 40.0) as usize)
        .map(|_| Blob {
            x: rng.uniform() * (w as f32 + 2.0 * margin) - margin,
            y: rng.uniform() * (h as f32 + 2.0 * margin) - margin,
            sigma: 2.0 + 4.0 * rng.uniform(),
            amp: 0.05 + 0.1 * rng.uniform(),
        })
        .collect()
}

/// Particle image pair with displacement `field` (pixels, as a function of position).
pub fn particle_pair(
    h: usize,
    w: usize,
    seed: u64,
    density: f32,
    sigma: f32,
    field: impl Fn(f32, f32) -> (f32, f32),
) -> (Frame, Frame) {
    let a = particles(h, w, seed, density, sigma);
    let b = displace(&a, field);
    (render(h, w, &a), render(h, w, &b))
}

/// Fog-texture image pair with displacement `field`.
pub fn fog_pair(
    h: usize,
    w: usize,
    seed: u64,
    field: impl Fn(f32, f32) -> (f32, f32),
) -> (Frame, Frame) {
    let a = fog(h, w, seed);
    let b = displace(&a, field);
    (render(h, w, &a), render(h, w, &b))
}

/// Adds zero-mean Gaussian noise with standard deviation `sigma`.
pub fn add_noise(frame: &mut Frame, sigma: f32, seed: u64) {
    let mut rng = Rng::new(seed);
    frame.mapv_inplace(|v| v + sigma * rng.normal());
}
