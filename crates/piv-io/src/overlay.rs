//! Text and chart overlays for quiver frames: a readout of the current pair's flow statistics
//! (top-left) and a chart of the whole run with a cursor at the current pair (bottom-right).

use std::path::Path;
use std::sync::LazyLock;

use ab_glyph::FontRef;
use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_antialiased_line_segment_mut, draw_text_mut, text_size};
use imageproc::pixelops::interpolate;
use piv_core::FlowStats;

use crate::flow::{EnergyUnits, FlowSeries};

/// DejaVu Sans Mono 2.37 (Bitstream Vera licence, `assets/DejaVuSansMono-LICENSE`).
static FONT: LazyLock<FontRef<'static>> = LazyLock::new(|| {
    FontRef::try_from_slice(include_bytes!("../assets/DejaVuSansMono.ttf")).expect("embedded font")
});

/// Vorticity colour (readout and chart).
pub const VORTICITY_COLOR: Rgb<u8> = Rgb([235, 235, 235]);
/// Kinetic energy colour. Distinct from the quiver colour map and the magenta of replaced vectors.
pub const KE_COLOR: Rgb<u8> = Rgb([80, 220, 170]);
const LABEL_COLOR: Rgb<u8> = Rgb([190, 190, 190]);
const CURSOR_COLOR: Rgb<u8> = Rgb([255, 255, 255]);
const GUIDE_COLOR: Rgb<u8> = Rgb([110, 110, 110]);
/// Remaining brightness of the image under a panel.
const PANEL_SHADE: f32 = 0.35;
/// Frames narrower than this get the chart in a strip below the image.
pub const MIN_OVERLAY_WIDTH: u32 = 480;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Where the overlays go on a `width × height` quiver image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    /// Output height: the image height, plus a strip for the chart on narrow frames.
    pub height: u32,
    pub chart: Rect,
    /// Readout text height, px.
    pub text_px: f32,
}

pub fn layout(width: u32, height: u32) -> Layout {
    let text_px = (height as f32 / 42.0).clamp(12.0, 26.0);
    let m = (text_px * 0.6).round() as u32;
    if width >= MIN_OVERLAY_WIDTH {
        let w = ((width as f32 * 0.35) as u32).max(360).min(width - 2 * m);
        let h = ((height as f32 * 0.24) as u32).max(180).min(height / 2);
        Layout {
            height,
            chart: Rect {
                x: width - m - w,
                y: height.saturating_sub(m + h),
                w,
                h,
            },
            text_px,
        }
    } else {
        let h = ((width as f32 * 0.5) as u32).max(170);
        Layout {
            height: height + h,
            chart: Rect {
                x: m.min(width / 8),
                y: height,
                w: width - 2 * m.min(width / 8),
                h: h - m.min(h / 8),
            },
            text_px,
        }
    }
}

/// Darkens `r` (clipped to the image).
fn shade(img: &mut RgbImage, r: Rect) {
    let (w, h) = img.dimensions();
    for y in r.y..(r.y + r.h).min(h) {
        for x in r.x..(r.x + r.w).min(w) {
            let p = img.get_pixel_mut(x, y);
            p.0 = p.0.map(|c| (c as f32 * PANEL_SHADE) as u8);
        }
    }
}

fn scaled(c: Rgb<u8>, k: f32) -> Rgb<u8> {
    Rgb(c.0.map(|v| (v as f32 * k) as u8))
}

fn text(img: &mut RgbImage, color: Rgb<u8>, x: i32, y: i32, px: f32, s: &str) {
    draw_text_mut(img, color, x, y, px, &*FONT, s);
}

fn text_w(px: f32, s: &str) -> u32 {
    text_size(px, &*FONT, s).0
}

/// Number with about three significant digits; scientific outside [0.01, 10⁴).
pub fn fmt_num(x: f64) -> String {
    if !x.is_finite() {
        return "–".into();
    }
    let a = x.abs();
    if a == 0.0 {
        "0".into()
    } else if (0.01..1e4).contains(&a) {
        let decimals = (2 - a.log10().floor() as i32).max(0) as usize;
        format!("{x:.decimals$}")
    } else {
        format!("{x:.2e}")
    }
}

fn fmt_signed(x: f64) -> String {
    let s = fmt_num(x);
    if x > 0.0 { format!("+{s}") } else { s }
}

/// One line of readout text and its colour.
pub type Line = (String, Rgb<u8>);

/// Readout lines for one pair. `title` names the panel in side-by-side videos.
pub fn readout(
    title: Option<&str>,
    t_s: f64,
    index: usize,
    count: usize,
    s: &FlowStats,
    units: &EnergyUnits,
) -> Vec<Line> {
    let mut lines = Vec::new();
    let head = format!("t {t_s:.4} s  pair {}/{count}", index + 1);
    lines.push((
        match title {
            Some(t) => format!("{t}  {head}"),
            None => head,
        },
        LABEL_COLOR,
    ));
    lines.push((
        format!(
            "ω mean {} 1/s  |ω| {} 1/s",
            fmt_signed(s.vorticity_mean as f64),
            fmt_num(s.vorticity_abs_mean as f64)
        ),
        VORTICITY_COLOR,
    ));
    let (total, unit) = units.ke_total(s.ke_sum);
    lines.push((
        format!(
            "KE {} {unit}  ({:.0}% measured)",
            fmt_num(total),
            100.0 * s.coverage
        ),
        KE_COLOR,
    ));
    let (mean, unit) = units.ke_mean(s.ke_mean);
    lines.push((format!("KE/mass {} {unit}", fmt_num(mean)), KE_COLOR));
    lines
}

/// Draws `lines` in a shaded box with its top-left corner at `(x, y)`.
pub fn draw_readout(img: &mut RgbImage, lines: &[Line], x: u32, y: u32, px: f32) {
    let pad = (px * 0.4).round() as u32;
    let lh = (px * 1.2).round() as u32;
    let w = lines.iter().map(|(s, _)| text_w(px, s)).max().unwrap_or(0) + 2 * pad;
    let h = lh * lines.len() as u32 + 2 * pad;
    shade(img, Rect { x, y, w, h });
    for (i, (s, c)) in lines.iter().enumerate() {
        text(
            img,
            *c,
            (x + pad) as i32,
            (y + pad + i as u32 * lh) as i32,
            px,
            s,
        );
    }
}

/// A series drawn in one strip of the chart.
pub struct Trace {
    pub values: Vec<f32>,
    pub color: Rgb<u8>,
    /// Main traces are drawn thick and bright up to the cursor; others thin and dim throughout.
    pub main: bool,
}

/// One panel of the chart, with its own y range.
pub struct Strip {
    pub title: String,
    pub traces: Vec<Trace>,
    /// Draw y = 0 and keep it in range.
    pub zero_line: bool,
}

impl Strip {
    fn range(&self) -> (f32, f32) {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in self
            .traces
            .iter()
            .flat_map(|t| &t.values)
            .filter(|v| v.is_finite())
        {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        if self.zero_line || !lo.is_finite() {
            lo = lo.min(0.0);
            hi = hi.max(0.0);
        }
        if hi - lo <= f32::EPSILON * hi.abs().max(1.0) {
            let d = (hi.abs() * 0.1).max(1.0);
            return (lo - d, hi + d);
        }
        let pad = 0.08 * (hi - lo);
        // Don't pad a non-negative quantity (energy) below zero.
        let floor = if lo >= 0.0 { 0.0 } else { f32::NEG_INFINITY };
        ((lo - pad).max(floor), hi + pad)
    }
}

/// Whole-run chart: strips stacked over a shared time axis.
pub struct Chart {
    pub t: Vec<f64>,
    pub strips: Vec<Strip>,
    ranges: Vec<(f32, f32)>,
}

impl Chart {
    /// Fixes each strip's y range from all of its values, so the chart doesn't rescale.
    pub fn new(t: Vec<f64>, strips: Vec<Strip>) -> Self {
        let ranges = strips.iter().map(Strip::range).collect();
        Self { t, strips, ranges }
    }

    /// The usual two strips: signed mean vorticity over total KE.
    pub fn vorticity_and_ke(
        t: Vec<f64>,
        vorticity: Vec<Vec<f32>>,
        ke_total: Vec<Vec<f32>>,
        ke_unit: &str,
    ) -> Self {
        let traces = |series: Vec<Vec<f32>>, color| {
            let n = series.len();
            series
                .into_iter()
                .enumerate()
                .map(|(i, values)| Trace {
                    values,
                    color,
                    main: i + 1 == n,
                })
                .collect()
        };
        Self::new(
            t,
            vec![
                Strip {
                    title: "ω mean (1/s, + = ccw)".into(),
                    traces: traces(vorticity, VORTICITY_COLOR),
                    zero_line: true,
                },
                Strip {
                    title: format!("KE ({ke_unit})"),
                    traces: traces(ke_total, KE_COLOR),
                    zero_line: false,
                },
            ],
        )
    }
}

/// Draws `chart` in `r` with a cursor at pair `cursor`. `px` is the readout text size; chart
/// labels are smaller.
pub fn draw_chart(img: &mut RgbImage, r: Rect, chart: &Chart, cursor: usize, px: f32) {
    shade(img, r);
    let n = chart.t.len();
    if n == 0 || chart.strips.is_empty() {
        return;
    }
    let lpx = (px * 0.72).max(10.0);
    let pad = (lpx * 0.6).round() as u32;
    let label_h = (lpx * 1.25).round() as u32;
    let x0 = (r.x + pad) as f32;
    let pw = r.w.saturating_sub(2 * pad).max(2) as f32;
    let (t0, t1) = (chart.t[0], chart.t[n - 1]);
    let xt = |i: usize| {
        let span = t1 - t0;
        let f = if span > 0.0 {
            (chart.t[i] - t0) / span
        } else {
            0.5
        };
        x0 + f as f32 * pw
    };
    let strips = chart.strips.len() as u32;
    let avail = r.h.saturating_sub(2 * pad + label_h);
    let sh = avail / strips;
    let cursor = cursor.min(n - 1);

    for (k, (strip, &(lo, hi))) in chart.strips.iter().zip(&chart.ranges).enumerate() {
        let top = r.y + pad + k as u32 * sh;
        let py0 = (top + label_h) as f32;
        let ph = sh.saturating_sub(label_h + pad / 2).max(2) as f32;
        let yv = |v: f32| py0 + (hi - v) / (hi - lo) * ph;
        text(img, LABEL_COLOR, x0 as i32, top as i32, lpx, &strip.title);
        let range = format!("{} … {}", fmt_num(lo as f64), fmt_num(hi as f64));
        let rw = text_w(lpx, &range);
        text(
            img,
            GUIDE_COLOR,
            (x0 + pw) as i32 - rw as i32,
            top as i32,
            lpx,
            &range,
        );
        if strip.zero_line {
            let y = yv(0.0).round() as i32;
            let mut x = x0 as i32;
            while x < (x0 + pw) as i32 {
                let e = (x + 6).min((x0 + pw) as i32);
                draw_antialiased_line_segment_mut(img, (x, y), (e, y), GUIDE_COLOR, interpolate);
                x += 12;
            }
        }
        for tr in strip
            .traces
            .iter()
            .filter(|t| !t.main)
            .chain(strip.traces.iter().filter(|t| t.main))
        {
            for i in 1..tr.values.len().min(n) {
                let (a, b) = (tr.values[i - 1], tr.values[i]);
                if !a.is_finite() || !b.is_finite() {
                    continue;
                }
                let (pa, pb) = ((xt(i - 1), yv(a)), (xt(i), yv(b)));
                let (color, offsets): (_, &[f32]) = if !tr.main {
                    (scaled(tr.color, 0.45), &[0.0])
                } else if i <= cursor {
                    (tr.color, &[0.0, 1.0])
                } else {
                    (scaled(tr.color, 0.4), &[0.0])
                };
                for &o in offsets {
                    let p = |q: (f32, f32)| (q.0.round() as i32, (q.1 + o).round() as i32);
                    draw_antialiased_line_segment_mut(img, p(pa), p(pb), color, interpolate);
                }
            }
        }
        // Cursor line through the plot area.
        let cx = xt(cursor).round() as i32;
        draw_antialiased_line_segment_mut(
            img,
            (cx, py0 as i32),
            (cx, (py0 + ph) as i32),
            CURSOR_COLOR,
            interpolate,
        );
    }
    // Time axis labels.
    let ty = (r.y + r.h).saturating_sub(pad + label_h) as i32 + (lpx * 0.2) as i32;
    text(img, GUIDE_COLOR, x0 as i32, ty, lpx, &format!("{t0:.3} s"));
    let end = format!("{t1:.3} s");
    let ew = text_w(lpx, &end);
    text(
        img,
        GUIDE_COLOR,
        (x0 + pw) as i32 - ew as i32,
        ty,
        lpx,
        &end,
    );
}

/// The chart for a run: signed mean vorticity over total KE in `units`. With several series
/// (same pairs), the last is drawn as the main trace and the others thin and dim behind it.
pub fn run_chart(series: &[&FlowSeries], units: &EnergyUnits) -> Chart {
    let t = series.last().map(|s| s.t.clone()).unwrap_or_default();
    let w = series
        .iter()
        .map(|s| s.column(|x| x.vorticity_mean))
        .collect();
    let e = series
        .iter()
        .map(|s| s.column(|x| units.ke_total(x.ke_sum).0 as f32))
        .collect();
    Chart::vorticity_and_ke(t, w, e, units.ke_total(0.0).1)
}

/// Rewrites the PNG at `path` with [`decorate`] applied.
pub fn decorate_png(path: &Path, lines: &[Line], chart: &Chart, cursor: usize) -> Result<()> {
    let img = image::open(path)
        .with_context(|| format!("reading {}", path.display()))?
        .to_rgb8();
    decorate(&img, lines, chart, cursor)
        .save(path)
        .with_context(|| format!("writing {}", path.display()))
}

/// Copies `img` onto a canvas of `layout.height` (adding the chart strip on narrow frames),
/// then draws the readout and chart.
pub fn decorate(img: &RgbImage, lines: &[Line], chart: &Chart, cursor: usize) -> RgbImage {
    let (w, h) = img.dimensions();
    let l = layout(w, h);
    let mut out = if l.height == h {
        img.clone()
    } else {
        let mut c = RgbImage::new(w, l.height);
        image::imageops::replace(&mut c, img, 0, 0);
        c
    };
    let m = (l.text_px * 0.6) as u32;
    draw_readout(&mut out, lines, m, m, l.text_px);
    draw_chart(&mut out, l.chart, chart, cursor, l.text_px);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ab_glyph::Font;

    fn stats() -> FlowStats {
        FlowStats {
            vorticity_mean: 1.5,
            vorticity_abs_mean: 4.0,
            ke_mean: 3000.0,
            ke_sum: 2.5e9,
            coverage: 0.8,
            n_usable: 100,
            n_vorticity: 90,
        }
    }

    fn chart(n: usize) -> Chart {
        let t = (0..n).map(|i| i as f64 * 0.005).collect();
        let w = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let e = (0..n).map(|i| i as f32).collect();
        Chart::vorticity_and_ke(t, vec![w], vec![e], "µJ")
    }

    #[test]
    fn font_has_every_glyph_used() {
        let units = [
            EnergyUnits {
                m_per_px: None,
                thickness_m: None,
                density: 1.2,
            },
            EnergyUnits {
                m_per_px: Some(1e-3),
                thickness_m: None,
                density: 1.2,
            },
            EnergyUnits {
                m_per_px: Some(1e-3),
                thickness_m: Some(1e-3),
                density: 1.2,
            },
        ];
        let mut all =
            String::from("ω mean (1/s, + = ccw) KE … – µJ mJ/m px⁴/s² m²/s² 0123456789.e+-%|");
        for u in &units {
            for (s, _) in readout(Some("raw"), 0.1, 0, 10, &stats(), u) {
                all += &s;
            }
        }
        for c in all.chars() {
            assert_ne!(FONT.glyph_id(c).0, 0, "no glyph for {c:?}");
        }
    }

    #[test]
    fn number_format() {
        assert_eq!(fmt_num(1.234), "1.23");
        assert_eq!(fmt_num(12.34), "12.3");
        assert_eq!(fmt_num(123.4), "123");
        assert_eq!(fmt_num(0.0123), "0.0123");
        assert_eq!(fmt_num(2.5e9), "2.50e9");
        assert_eq!(fmt_num(0.0), "0");
        assert_eq!(fmt_num(f64::NAN), "–");
        assert_eq!(fmt_signed(1.5), "+1.50");
        assert_eq!(fmt_signed(-1.5), "-1.50");
    }

    #[test]
    fn overlays_stay_in_their_boxes() {
        let (w, h) = (1280, 720);
        let img = RgbImage::from_pixel(w, h, Rgb([100, 100, 100]));
        let l = layout(w, h);
        assert_eq!(l.height, h);
        let lines = readout(
            None,
            0.1,
            3,
            50,
            &stats(),
            &EnergyUnits {
                m_per_px: None,
                thickness_m: None,
                density: 1.2,
            },
        );
        let out = decorate(&img, &lines, &chart(50), 20);
        assert_eq!(out.dimensions(), (w, h));
        // Corners of the readout and the chart are shaded.
        let m = (l.text_px * 0.6) as u32;
        assert!(out.get_pixel(m + 1, m + 1).0[0] < 100);
        let c = l.chart;
        assert!(out.get_pixel(c.x + 1, c.y + c.h - 2).0[0] < 100);
        // The middle of the frame, where the vectors are, is untouched.
        for (x, y) in [(w / 2, h / 2), (w / 4, h * 3 / 4), (w * 3 / 4, h / 4)] {
            assert_eq!(out.get_pixel(x, y).0, [100, 100, 100], "({x}, {y})");
        }
        // Something was drawn in the chart (traces brighter than the shaded panel).
        let lit = (c.x..c.x + c.w)
            .flat_map(|x| (c.y..c.y + c.h).map(move |y| (x, y)))
            .filter(|&(x, y)| out.get_pixel(x, y).0[1] > 150)
            .count();
        assert!(lit > 100, "{lit}");
    }

    #[test]
    fn narrow_frames_get_a_strip_below() {
        let (w, h) = (320, 240);
        let img = RgbImage::from_pixel(w, h, Rgb([100, 100, 100]));
        let l = layout(w, h);
        assert!(l.height > h);
        assert!(l.chart.y >= h && l.chart.y + l.chart.h <= l.height);
        assert!(l.chart.x + l.chart.w <= w);
        let out = decorate(&img, &[], &chart(10), 0);
        assert_eq!(out.dimensions(), (w, l.height));
        assert_eq!(out.get_pixel(w - 2, h - 2).0, [100, 100, 100]);
    }

    #[test]
    fn flat_and_empty_series_have_a_range() {
        let c = Chart::vorticity_and_ke(
            vec![0.0, 1.0],
            vec![vec![f32::NAN; 2]],
            vec![vec![5.0; 2]],
            "x",
        );
        for &(lo, hi) in &c.ranges {
            assert!(lo.is_finite() && hi.is_finite() && hi > lo);
        }
        let c = chart(50);
        assert_eq!(c.ranges[1].0, 0.0, "KE axis starts at 0, not below");
        assert!(c.ranges[0].0 < -1.0, "vorticity axis keeps its padding");
        let mut img = RgbImage::new(600, 400);
        draw_chart(
            &mut img,
            Rect {
                x: 0,
                y: 0,
                w: 600,
                h: 400,
            },
            &c,
            5,
            16.0,
        );
    }
}
