use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use ndarray::{Array3, Axis, Zip};
use piv_core::smooth::{self, TemporalKernel};
use piv_core::{Grid, VectorField, flags};
use piv_io::decode::{Channel, DecodeOptions, FfmpegSource, FrameSource, Roi};
use piv_io::probe;
use piv_io::render::{QuiverStyle, RgbImage, encode_video, render_quiver};
use piv_io::run_file::{RunData, SmoothedFields, read_run, write_smoothed};
use serde_json::json;

use crate::png::PngSink;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KernelArg {
    Mean,
    Median,
}

#[derive(Debug, Args)]
pub struct SmoothArgs {
    /// HDF5 file written by `piv analyze`.
    input: PathBuf,
    /// Output HDF5 file (smoothed fields plus the raw fields under /raw).
    #[arg(short, long, required_unless_present = "sweep")]
    output: Option<PathBuf>,
    /// Overwrite existing output.
    #[arg(long)]
    force: bool,
    /// Temporal window in pairs (odd). At a pair every frame and 1000 fps, 9 pairs ≈ 9 ms.
    #[arg(long)]
    temporal: Option<usize>,
    /// Temporal filter: median resists leftover outliers; mean removes more random noise.
    #[arg(long, value_enum, default_value_t = KernelArg::Median)]
    temporal_kernel: KernelArg,
    /// Spatial Gaussian sigma, in grid cells (vector spacing).
    #[arg(long)]
    spatial: Option<f32>,
    /// Minimum raw vectors in the temporal window for an output (default: a third of it).
    #[arg(long)]
    min_support: Option<usize>,
    /// Compare a range of settings (cross-validation table) instead of writing output.
    #[arg(long)]
    sweep: bool,
    /// Write side-by-side raw | smoothed quiver PNGs and an MP4 (re-decodes the source video).
    #[arg(long)]
    video: bool,
    /// Playback rate of the comparison video.
    #[arg(long, default_value_t = 30.0)]
    video_fps: f64,
    /// Draw every n-th vector.
    #[arg(long, default_value_t = 1)]
    quiver_every: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Settings {
    temporal: Option<(TemporalKernel, usize)>,
    spatial: Option<f32>,
    min_support: usize,
}

impl Settings {
    fn label(&self) -> String {
        let t = self
            .temporal
            .map(|(k, w)| format!("temporal {k:?} {w}").to_lowercase());
        let s = self.spatial.map(|s| format!("spatial σ={s}"));
        match (t, s) {
            (Some(t), Some(s)) => format!("{t} + {s}"),
            (Some(t), None) => t,
            (None, Some(s)) => s,
            (None, None) => "raw".into(),
        }
    }
}

/// Inputs shared by every filter run.
struct Prepared<'a> {
    run: &'a RunData,
    /// Raw vectors that may contribute: measured and not rejected.
    usable: Array3<bool>,
    /// Where an output is wanted: the window had seeding in that pair.
    gate: Array3<bool>,
    /// Raw displacements with unusable vectors set to NaN.
    raw_dx: Array3<f32>,
    raw_dy: Array3<f32>,
    /// Pair offsets that share a frame with the centre pair (left out in cross-validation).
    shared_frames: Vec<isize>,
    /// Grid distance within which windows overlap (left out in cross-validation).
    overlap_radius: usize,
}

impl<'a> Prepared<'a> {
    fn new(run: &'a RunData) -> Self {
        let usable = Zip::from(&run.flags)
            .and(&run.dx)
            .and(&run.dy)
            .map_collect(|&f, u, v| f & flags::INVALID == 0 && u.is_finite() && v.is_finite());
        let gate = run.flags.mapv(|f| f & flags::NO_SIGNAL == 0);
        let raw_dx = smooth::mask(&run.dx, &usable);
        let raw_dy = smooth::mask(&run.dy, &usable);

        // Pairs t and t+k share a frame when frame_b of one is frame_a of the other.
        let stride = run.meta["stride"].as_u64().unwrap_or(1) as i64;
        let step = run.meta["pair_step"].as_u64().unwrap_or(1) as i64;
        let mut shared_frames = vec![0];
        if stride % step == 0 {
            shared_frames.extend([stride / step, -stride / step].map(|k| k as isize));
        }
        // Windows k grid steps apart overlap while k·step < window.
        let overlap = run.meta["piv"]["overlap"].as_f64().unwrap_or(0.0);
        let overlap_radius = ((1.0 / (1.0 - overlap)).ceil() as usize).saturating_sub(1);
        Self {
            run,
            usable,
            gate,
            raw_dx,
            raw_dy,
            shared_frames,
            overlap_radius,
        }
    }

    /// Applies the filters. With `leave_out`, each output is predicted without the raw vector
    /// at that point (for cross-validation).
    fn apply(&self, s: &Settings, leave_out: bool) -> (Array3<f32>, Array3<f32>, Array3<u16>) {
        let (mut dx, mut dy, support) = match s.temporal {
            Some((kernel, w)) => {
                // When a spatial pass follows, it does the leaving out: it skips the centre
                // point (whose temporal value includes the raw vector) and overlapping windows.
                let excl: &[isize] = if leave_out && s.spatial.is_none() {
                    &self.shared_frames
                } else {
                    &[]
                };
                let t = smooth::temporal(
                    &self.run.dx,
                    &self.run.dy,
                    &self.usable,
                    &self.gate,
                    kernel,
                    w,
                    s.min_support,
                    excl,
                );
                (t.dx, t.dy, t.support)
            }
            None => (
                self.raw_dx.clone(),
                self.raw_dy.clone(),
                self.usable.mapv(u16::from),
            ),
        };
        if let Some(sigma) = s.spatial {
            let excl = leave_out.then_some(self.overlap_radius);
            (dx, dy) = smooth::spatial(&dx, &dy, &self.gate, sigma, 0.25, excl);
        }
        (dx, dy, support)
    }

    fn report(&self, s: &Settings) -> Report {
        let (dx, dy, _) = self.apply(s, false);
        let (cv, cv_n) = if s.temporal.is_none() && s.spatial.is_none() {
            (f32::NAN, 0)
        } else {
            let (px, py, _) = self.apply(s, true);
            smooth::cross_validation_error(&self.raw_dx, &self.raw_dy, &self.usable, &px, &py)
        };
        Report::new(s.label(), &dx, &dy, &self.gate, &self.usable, cv, cv_n)
    }
}

struct Report {
    label: String,
    cv: f32,
    cv_n: usize,
    coverage: f64,
    p50: f32,
    p95: f32,
    change: f32,
    roughness: f32,
}

impl Report {
    fn new(
        label: String,
        dx: &Array3<f32>,
        dy: &Array3<f32>,
        gate: &Array3<bool>,
        usable: &Array3<bool>,
        cv: f32,
        cv_n: usize,
    ) -> Self {
        let seeded = gate.iter().filter(|&&g| g).count();
        let defined = dx
            .iter()
            .zip(dy)
            .filter(|(u, v)| u.is_finite() && v.is_finite())
            .count();
        let coverage = defined as f64 / seeded.max(1) as f64;
        // Compare magnitudes on the same points for every setting: where raw was accepted.
        let mut mags: Vec<f32> =
            Zip::from(dx)
                .and(dy)
                .and(usable)
                .fold(Vec::new(), |mut acc, u, v, &ok| {
                    let m = u.hypot(*v);
                    if ok && m.is_finite() {
                        acc.push(m);
                    }
                    acc
                });
        mags.sort_unstable_by(f32::total_cmp);
        let q = |p: f64| {
            mags.get(((mags.len().max(1) - 1) as f64 * p) as usize)
                .copied()
                .unwrap_or(f32::NAN)
        };
        Self {
            label,
            cv,
            cv_n,
            coverage,
            p50: q(0.5),
            p95: q(0.95),
            change: smooth::pair_to_pair_change(dx, dy),
            roughness: smooth::spatial_roughness(dx, dy),
        }
    }

    fn header() {
        println!(
            "{:<34} {:>9} {:>9} {:>8} {:>8} {:>10} {:>10}",
            "setting", "CV err", "coverage", "|d| p50", "|d| p95", "pair→pair", "roughness"
        );
    }

    fn row(&self) {
        let cv = if self.cv.is_finite() {
            format!("{:.3}", self.cv)
        } else {
            "–".into()
        };
        println!(
            "{:<34} {:>9} {:>8.1}% {:>8.2} {:>8.2} {:>10.3} {:>10.3}",
            self.label,
            cv,
            100.0 * self.coverage,
            self.p50,
            self.p95,
            self.change,
            self.roughness
        );
    }
}

fn explain() {
    println!(
        "\nAll values in px of displacement per pair.\n\
         CV err     median distance between each raw vector and its prediction from neighbours that share no \
         data with it\n           (overlapping windows and pairs sharing a frame are left out) — lower is better; \
         it cannot go below the raw measurement noise.\n\
         coverage   seeded grid points with a value (raw: accepted vectors only).\n\
         |d| p50/95 at raw-accepted points; a falling p95 means peaks are being flattened (smoothing bias).\n\
         pair→pair  median change between consecutive pairs (temporal jitter).\n\
         roughness  median deviation from the mean of 4 neighbours (spatial jitter)."
    );
}

pub fn run(args: SmoothArgs) -> Result<()> {
    let run = read_run(&args.input)?;
    ensure!(
        !run.is_empty(),
        "{} contains no pairs",
        args.input.display()
    );
    let prep = Prepared::new(&run);
    let kernel = match args.temporal_kernel {
        KernelArg::Mean => TemporalKernel::Mean,
        KernelArg::Median => TemporalKernel::Median,
    };
    let (nt, ny, nx) = run.dx.dim();
    let pair_step = run.meta["pair_step"].as_u64().unwrap_or(1) as f64;
    let fps = run.meta["capture_fps"].as_f64().unwrap_or(f64::NAN);
    println!(
        "input:   {} — {nt} pairs, {nx}×{ny} grid, dt {:.3} ms, a pair every {:.3} ms",
        args.input.display(),
        run.dt()? * 1e3,
        pair_step / fps * 1e3
    );

    if args.sweep {
        return sweep(&prep, args.min_support);
    }

    if let Some(w) = args.temporal {
        ensure!(w >= 3 && w % 2 == 1, "--temporal must be an odd number ≥ 3");
    }
    if let Some(s) = args.spatial {
        ensure!(s > 0.0, "--spatial must be positive");
    }
    if args.temporal.is_none() && args.spatial.is_none() {
        bail!(
            "nothing to do: pass --temporal N and/or --spatial SIGMA (or --sweep to compare settings)"
        );
    }
    let settings = Settings {
        temporal: args.temporal.map(|w| (kernel, w)),
        spatial: args.spatial,
        min_support: args
            .min_support
            .unwrap_or_else(|| args.temporal.map_or(1, |w| w.div_ceil(3))),
    };

    let output = args
        .output
        .as_ref()
        .expect("clap requires --output without --sweep");
    if output.exists() && !args.force {
        bail!("{} exists; pass --force to overwrite", output.display());
    }
    if output.canonicalize().ok() == args.input.canonicalize().ok() {
        bail!("output would overwrite the input");
    }

    Report::header();
    let raw = prep.report(&Settings {
        temporal: None,
        spatial: None,
        min_support: 1,
    });
    raw.row();
    let rep = prep.report(&settings);
    rep.row();
    explain();

    let (dx, dy, support) = prep.apply(&settings, false);
    let span_ms = settings
        .temporal
        .map(|(_, w)| w as f64 * pair_step / fps * 1e3);
    let settings_json = json!({
        "smooth_source": args.input.display().to_string(),
        "smooth_temporal_window": settings.temporal.map(|(_, w)| w),
        "smooth_temporal_kernel": settings.temporal.map(|(k, _)| format!("{k:?}").to_lowercase()),
        "smooth_temporal_span_ms": span_ms,
        "smooth_spatial_sigma_cells": settings.spatial,
        "smooth_spatial_min_weight": settings.spatial.map(|_| 0.25),
        "smooth_min_support": settings.min_support,
        "smooth_cv_median_px": rep.cv,
        "smooth_cv_vectors": rep.cv_n,
    });
    write_smoothed(
        output,
        &run,
        &SmoothedFields {
            dx: &dx,
            dy: &dy,
            support: &support,
            settings: settings_json,
        },
    )?;
    println!("\nwrote {}", output.display());

    if args.video {
        comparison_video(
            &run,
            &prep,
            &dx,
            &dy,
            output,
            args.video_fps,
            args.quiver_every,
            args.force,
        )?;
    }
    Ok(())
}

fn sweep(prep: &Prepared, min_support: Option<usize>) -> Result<()> {
    let ms = |w: usize| min_support.unwrap_or(w.div_ceil(3));
    let mut configs = vec![Settings {
        temporal: None,
        spatial: None,
        min_support: 1,
    }];
    for kernel in [TemporalKernel::Median, TemporalKernel::Mean] {
        for w in [3, 5, 9, 15, 25] {
            configs.push(Settings {
                temporal: Some((kernel, w)),
                spatial: None,
                min_support: ms(w),
            });
        }
    }
    for s in [0.5, 1.0, 1.5, 2.0] {
        configs.push(Settings {
            temporal: None,
            spatial: Some(s),
            min_support: 1,
        });
    }
    for w in [5, 9, 15] {
        for s in [0.5, 1.0] {
            configs.push(Settings {
                temporal: Some((TemporalKernel::Median, w)),
                spatial: Some(s),
                min_support: ms(w),
            });
        }
    }
    Report::header();
    let mut best: Option<Report> = None;
    for c in &configs {
        let r = prep.report(c);
        r.row();
        if r.cv.is_finite() && best.as_ref().is_none_or(|b| r.cv < b.cv) {
            best = Some(r);
        }
    }
    explain();
    if let Some(b) = best {
        println!("\nlowest CV error: {} ({:.3} px)", b.label, b.cv);
    }
    Ok(())
}

/// Re-creates the final-pass grid from the run metadata (needed to draw vectors on frames).
fn run_grid(run: &RunData) -> Result<(Grid, Option<Roi>)> {
    let m = &run.meta;
    let roi: Option<Roi> = serde_json::from_value(m["roi"].clone()).context("reading roi")?;
    let (h, w) = match roi {
        Some(r) => (r.height, r.width),
        None => (
            m["frame_height"].as_u64().context("frame_height")? as usize,
            m["frame_width"].as_u64().context("frame_width")? as usize,
        ),
    };
    let win = m["piv"]["windows"]
        .as_array()
        .and_then(|a| a.last())
        .and_then(|v| v.as_u64())
        .context("windows")?;
    let overlap = m["piv"]["overlap"].as_f64().context("overlap")? as f32;
    let grid = Grid::new(h, w, win as usize, overlap)?;
    let ox = roi.map_or(0.0, |r| r.x as f32);
    ensure!(
        grid.nx == run.x.len() && (grid.center_x(0) + ox - run.x[0]).abs() < 1e-3,
        "could not reconstruct the analysis grid from the run metadata"
    );
    Ok((grid, roi))
}

fn field(
    grid: &Grid,
    dx: &Array3<f32>,
    dy: &Array3<f32>,
    fl: &Array3<u8>,
    t: usize,
) -> VectorField {
    let mut f = VectorField::new(grid.clone());
    f.dx = dx.index_axis(Axis(0), t).to_owned();
    f.dy = dy.index_axis(Axis(0), t).to_owned();
    f.flags = fl.index_axis(Axis(0), t).to_owned();
    f
}

/// 75th and 95th percentile of accepted raw displacement magnitudes over the whole run.
fn run_percentiles(prep: &Prepared) -> (f32, f32) {
    let mut m: Vec<f32> = Zip::from(&prep.raw_dx)
        .and(&prep.raw_dy)
        .and(&prep.usable)
        .fold(Vec::new(), |mut acc, u, v, &ok| {
            if ok {
                acc.push(u.hypot(*v));
            }
            acc
        });
    if m.is_empty() {
        return (1.0, 1.0);
    }
    m.sort_unstable_by(f32::total_cmp);
    let q = |p: f64| m[((m.len() - 1) as f64 * p) as usize];
    (q(0.75), q(0.95))
}

#[allow(clippy::too_many_arguments)]
fn comparison_video(
    run: &RunData,
    prep: &Prepared,
    dx: &Array3<f32>,
    dy: &Array3<f32>,
    output: &Path,
    fps: f64,
    every: usize,
    force: bool,
) -> Result<()> {
    let (grid, roi) = run_grid(run)?;
    let source = PathBuf::from(run.meta["source_path"].as_str().context("source_path")?);
    let info =
        probe(&source).with_context(|| format!("re-opening source video {}", source.display()))?;
    let channel: Channel = serde_json::from_value(run.meta["channel"].clone()).unwrap_or_default();

    let stem = output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "smooth".into());
    let dir = output.with_file_name(format!("{stem}_compare"));
    if dir.exists() {
        ensure!(force, "{} exists; pass --force to overwrite", dir.display());
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir_all(&dir)?;

    let first = *run.frame_a.iter().min().expect("non-empty") as usize;
    let last = *run.frame_a.iter().max().expect("non-empty") as usize;
    let opts = DecodeOptions {
        start: first,
        count: Some(last + 1 - first),
        channel,
        roi,
        high_bit_depth: info.bit_depth > 8,
        container_rate: info.container_rate,
        frame_size: (info.height, info.width),
    };
    let by_frame: std::collections::HashMap<usize, usize> = run
        .frame_a
        .iter()
        .enumerate()
        .map(|(t, &a)| (a as usize, t))
        .collect();
    let smoothed_flags = dx.mapv(|d| if d.is_finite() { 0 } else { flags::NO_SIGNAL });

    let mut src = FfmpegSource::open(&source, &opts)?;
    let pngs = PngSink::spawn();
    let mut style: Option<QuiverStyle> = None;
    while let Some((idx, frame)) = src.next_frame()? {
        let Some(&t) = by_frame.get(&idx) else {
            continue;
        };
        let raw = field(&grid, &run.dx, &run.dy, &run.flags, t);
        let sm = field(&grid, dx, dy, &smoothed_flags, t);
        let s = *style.get_or_insert_with(|| {
            // Scale arrows from the whole run, not the first pair, so quiet and busy moments
            // are both readable and arrow lengths are comparable across the video.
            let auto = QuiverStyle::auto(&frame, &raw);
            let (p75, p95) = run_percentiles(prep);
            QuiverStyle {
                arrow_scale: 0.6 * grid.step as f32 / p75.max(0.1),
                color_max: p95.max(0.1),
                every,
                ..auto
            }
        });
        let (left, right) = (
            render_quiver(&frame, &raw, &s),
            render_quiver(&frame, &sm, &s),
        );
        let (w, h) = left.dimensions();
        let mut both = RgbImage::new(2 * w + 8, h);
        image_copy(&mut both, &left, 0);
        image_copy(&mut both, &right, w + 8);
        pngs.send(dir.join(format!("compare_f{idx:06}.png")), both);
    }
    let n = pngs.finish()?;
    let video = output.with_file_name(format!("{stem}_compare.mp4"));
    encode_video(&dir, &video, fps)?;
    println!(
        "compare: {n} PNGs in {} (left raw, right smoothed)\nvideo:   {}",
        dir.display(),
        video.display()
    );
    Ok(())
}

fn image_copy(dst: &mut RgbImage, src: &RgbImage, x0: u32) {
    for (x, y, p) in src.enumerate_pixels() {
        dst.put_pixel(x0 + x, y, *p);
    }
}
