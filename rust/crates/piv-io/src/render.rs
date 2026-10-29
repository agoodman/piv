//! Quiver overlays: vectors drawn over the analysed frame, and MP4 assembly via ffmpeg.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use image::Rgb;
pub use image::RgbImage;
use imageproc::drawing::draw_antialiased_line_segment_mut;
use imageproc::pixelops::interpolate;
use piv_core::{Frame, VectorField, flags};

/// Rendering parameters. Keep them fixed across a run so arrow lengths and colours are
/// comparable between frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuiverStyle {
    /// Arrow length in image pixels per pixel of displacement.
    pub arrow_scale: f32,
    /// Displacement magnitude (px) at the top of the colour map.
    pub color_max: f32,
    /// Multiplier applied to the background image intensity.
    pub gain: f32,
    /// Draw every n-th vector in each direction.
    pub every: usize,
}

impl QuiverStyle {
    /// Scales arrows so a typical moving vector (75th percentile) spans ~60 % of the grid
    /// spacing, colours up to the 95th percentile, and stretches the background so its 99th
    /// percentile is near white.
    pub fn auto(raw: &Frame, field: &VectorField) -> Self {
        let mut mags: Vec<f32> = field
            .dx
            .iter()
            .zip(&field.dy)
            .zip(&field.flags)
            .filter(|&(_, &f)| f & flags::INVALID == 0)
            .map(|((u, v), _)| u.hypot(*v))
            .filter(|m| m.is_finite())
            .collect();
        let m75 = percentile(&mut mags, 0.75)
            .filter(|&m| m > 0.1)
            .unwrap_or(1.0);
        let m95 = percentile(&mut mags, 0.95)
            .filter(|&m| m > 0.1)
            .unwrap_or(1.0);
        let mut px: Vec<f32> = raw.iter().step_by(7).copied().collect();
        let p99 = percentile(&mut px, 0.99)
            .filter(|&p| p > 1e-3)
            .unwrap_or(1.0);
        Self {
            arrow_scale: 0.6 * field.grid.step as f32 / m75,
            color_max: m95,
            gain: (0.9 / p99).clamp(1.0, 20.0),
            every: 1,
        }
    }
}

fn percentile(v: &mut [f32], q: f32) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    let k = ((v.len() - 1) as f32 * q).round() as usize;
    Some(*v.select_nth_unstable_by(k, f32::total_cmp).1)
}

/// Light blue → yellow → red: stays visible on a dark, green-tinted background.
fn colormap(t: f32) -> Rgb<u8> {
    const STOPS: [(f32, [f32; 3]); 3] = [
        (0.0, [90.0, 170.0, 255.0]),
        (0.5, [255.0, 230.0, 80.0]),
        (1.0, [255.0, 70.0, 50.0]),
    ];
    let t = t.clamp(0.0, 1.0);
    let i = if t < 0.5 { 0 } else { 1 };
    let (t0, c0) = STOPS[i];
    let (t1, c1) = STOPS[i + 1];
    let s = (t - t0) / (t1 - t0);
    Rgb(std::array::from_fn(|k| {
        (c0[k] + s * (c1[k] - c0[k])).round() as u8
    }))
}

/// Colour for vectors that were rejected and replaced by interpolation.
pub const REPLACED_COLOR: Rgb<u8> = Rgb([255, 0, 255]);

/// Draws `field` over `raw` (the analysed, possibly ROI-cropped, frame). Missing vectors are
/// not drawn; replaced vectors are drawn in [`REPLACED_COLOR`].
pub fn render_quiver(raw: &Frame, field: &VectorField, style: &QuiverStyle) -> RgbImage {
    let (h, w) = raw.dim();
    let mut img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let g = ((raw[[y as usize, x as usize]] * style.gain).min(1.0) * 180.0) as u8;
        Rgb([g, g, g])
    });
    let thick = w >= 1280;
    let (xs, ys) = (field.xs(), field.ys());
    for (iy, &y) in ys.iter().enumerate().step_by(style.every.max(1)) {
        for (ix, &x) in xs.iter().enumerate().step_by(style.every.max(1)) {
            let (u, v, f) = (
                field.dx[[iy, ix]],
                field.dy[[iy, ix]],
                field.flags[[iy, ix]],
            );
            if !u.is_finite() || !v.is_finite() {
                continue;
            }
            let color = if f & flags::REPLACED != 0 {
                REPLACED_COLOR
            } else {
                colormap(u.hypot(v) / style.color_max)
            };
            arrow(
                &mut img,
                (x, y),
                (x + u * style.arrow_scale, y + v * style.arrow_scale),
                color,
                thick,
            );
        }
    }
    img
}

fn arrow(img: &mut RgbImage, from: (f32, f32), to: (f32, f32), color: Rgb<u8>, thick: bool) {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let len = dx.hypot(dy);
    let mut segs = vec![(from, to)];
    if len >= 2.0 {
        let head = (0.3 * len).clamp(2.0, 8.0);
        let (ux, uy) = (dx / len, dy / len);
        for side in [-1.0f32, 1.0] {
            let (c, s) = (0.42f32.cos(), side * 0.42f32.sin()); // ±24°
            let (bx, by) = (-(ux * c - uy * s), -(ux * s + uy * c));
            segs.push((to, (to.0 + head * bx, to.1 + head * by)));
        }
    }
    let (px, py) = if len > 0.0 {
        (-dy / len, dx / len)
    } else {
        (0.0, 0.0)
    };
    let offsets: &[f32] = if thick { &[0.0, 0.7] } else { &[0.0] };
    for &o in offsets {
        for &(a, b) in &segs {
            let p = |q: (f32, f32)| ((q.0 + o * px).round() as i32, (q.1 + o * py).round() as i32);
            draw_antialiased_line_segment_mut(img, p(a), p(b), color, interpolate);
        }
    }
}

/// Encodes every `*.png` in `dir` (in name order) into an H.264 MP4 at `fps`.
pub fn encode_video(dir: &Path, out: &Path, fps: f64) -> Result<()> {
    let pattern = dir.join("*.png");
    let status = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-framerate",
            &format!("{fps}"),
        ])
        .args(["-pattern_type", "glob", "-i"])
        .arg(&pattern)
        .args([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ])
        .arg(out)
        .status()
        .context("failed to run ffmpeg")?;
    if !status.success() {
        bail!("ffmpeg failed to encode {}", out.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use piv_core::Grid;

    #[test]
    fn draws_arrows_where_vectors_exist() {
        let raw = Frame::zeros((64, 64));
        let mut f = VectorField::new(Grid::new(64, 64, 32, 0.0).unwrap());
        f.dx.fill(4.0);
        f.dy.fill(0.0);
        f.flags.fill(0);
        f.dx[[1, 1]] = f32::NAN; // missing: not drawn
        f.flags[[1, 1]] = flags::NO_SIGNAL;
        let style = QuiverStyle::auto(&raw, &f);
        assert!((style.arrow_scale - 0.6 * 32.0 / 4.0).abs() < 1e-4);
        // Short arrows so each stays inside its own cell.
        let style = QuiverStyle {
            arrow_scale: 2.0,
            ..style
        };
        let img = render_quiver(&raw, &f, &style);
        let lit = |x0: u32, y0: u32| {
            (x0..x0 + 32).any(|x| (y0..y0 + 32).any(|y| img.get_pixel(x, y).0 != [0, 0, 0]))
        };
        assert!(lit(0, 0) && lit(32, 0) && lit(0, 32));
        assert!(!lit(32, 32));
    }

    #[test]
    fn colormap_endpoints() {
        assert_eq!(colormap(0.0), Rgb([90, 170, 255]));
        assert_eq!(colormap(2.0), Rgb([255, 70, 50]));
    }
}
