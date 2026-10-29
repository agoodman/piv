use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use piv_core::flowstats::neighbour_noise;
use piv_core::preprocess::{BackgroundEstimator, BackgroundMode};
use piv_core::validate::ValidationConfig;
use piv_core::{
    Frame, FrameRate, MaskConfig, PivAnalyzer, PivConfig, PreparedFrame, PreprocessConfig,
    VectorField, flags,
};
use piv_io::decode::{DecodeOptions, FfmpegSource, FrameSource, read_frame};
use piv_io::flow::{AIR_DENSITY, EnergyUnits, FlowSeries, pair_stats, usable};
use piv_io::overlay::{decorate_png, readout, run_chart};
use piv_io::render::{QuiverStyle, encode_video, render_quiver};
use piv_io::write::{FieldWriter, OutputFormat, PairRecord, RunMetadata, create_writer};
use piv_io::{Channel, FovAxis, MediaInfo, Roi, ScaleSpec, probe, resolve_capture_rate};
use rayon::prelude::*;

use crate::png::PngSink;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ChannelArg {
    /// FFmpeg's gray conversion (mixes in chroma; see piv_io::decode::Channel).
    Luma,
    /// Green after conversion to RGB.
    Green,
    /// The stored Y plane, unconverted (what PIVKit reads).
    Yplane,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FormatArg {
    Hdf5,
    Npz,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AxisArg {
    Diagonal,
    Horizontal,
    Vertical,
}

#[derive(Debug, Args)]
pub struct AnalyzeArgs {
    /// Input video.
    input: PathBuf,
    /// Output path: an .h5 file (HDF5) or a directory (NPZ).
    #[arg(short, long)]
    output: PathBuf,
    /// Overwrite existing output.
    #[arg(long)]
    force: bool,

    // ── Frame selection ─────────────────────────────────────────────────────
    /// First frame index to use.
    #[arg(long, conflicts_with = "start_s", help_heading = "Frames")]
    start: Option<usize>,
    /// Frame index to stop before (exclusive). Default: end of clip.
    #[arg(long, conflicts_with = "end_s", help_heading = "Frames")]
    end: Option<usize>,
    /// First frame, as capture-time seconds.
    #[arg(long, help_heading = "Frames")]
    start_s: Option<f64>,
    /// End, as capture-time seconds (exclusive).
    #[arg(long, help_heading = "Frames")]
    end_s: Option<f64>,
    /// Frame separation within a pair: frame i is paired with i + stride.
    #[arg(long, default_value_t = 1, help_heading = "Frames")]
    stride: usize,
    /// Spacing between successive pairs (1 = a pair starting at every frame).
    #[arg(long, default_value_t = 1, help_heading = "Frames")]
    pair_step: usize,
    /// Capture frame rate override (e.g. 240, 1000, 239.76, 240000/1001).
    #[arg(long, help_heading = "Frames")]
    capture_fps: Option<FrameRate>,

    // ── Image ───────────────────────────────────────────────────────────────
    /// Image plane to analyse.
    #[arg(long, value_enum, default_value_t = ChannelArg::Luma, help_heading = "Image")]
    channel: ChannelArg,
    /// Region of interest, x,y,w,h in full-frame pixels.
    #[arg(long, help_heading = "Image")]
    roi: Option<Roi>,
    /// Static background subtraction: none, min:N or mean:N (N frames sampled evenly across
    /// the whole clip, so slowly moving fog is not mistaken for background).
    #[arg(long, default_value = "min:40", help_heading = "Image")]
    background: String,
    /// Gaussian smoothing sigma in px (0 = off).
    #[arg(long, default_value_t = PreprocessConfig::default().smooth_sigma, help_heading = "Image")]
    smooth: f32,
    /// Local contrast normalization radius in px (0 = off).
    #[arg(long, default_value_t = PreprocessConfig::default().normalize_radius, help_heading = "Image")]
    normalize_radius: usize,

    // ── Interrogation ───────────────────────────────────────────────────────
    /// Window sizes per pass, largest first.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "96,64",
        help_heading = "Interrogation"
    )]
    window: Vec<usize>,
    /// Window overlap fraction.
    #[arg(long, default_value_t = 0.5, help_heading = "Interrogation")]
    overlap: f32,
    /// Number of passes at the final window size.
    #[arg(long, default_value_t = 2, help_heading = "Interrogation")]
    final_iterations: usize,
    /// Minimum window mean intensity after background subtraction (0–1).
    #[arg(long, default_value_t = 0.04, help_heading = "Interrogation")]
    min_mean: f32,
    /// Minimum window intensity standard deviation (0–1).
    #[arg(long, default_value_t = 0.015, help_heading = "Interrogation")]
    min_std: f32,

    // ── Validation ──────────────────────────────────────────────────────────
    /// Minimum primary/secondary correlation peak ratio.
    #[arg(long, default_value_t = ValidationConfig::default().ppr_threshold, help_heading = "Validation")]
    ppr: f32,
    /// Normalized median test threshold.
    #[arg(long, default_value_t = ValidationConfig::default().median_threshold, help_heading = "Validation")]
    median_threshold: f32,
    /// Normalized median test noise level ε (px).
    #[arg(long, default_value_t = ValidationConfig::default().median_eps, help_heading = "Validation")]
    median_eps: f32,
    /// Keep rejected vectors as NaN instead of replacing them.
    #[arg(long, help_heading = "Validation")]
    no_replace: bool,

    // ── Output ──────────────────────────────────────────────────────────────
    #[arg(long, value_enum, default_value_t = FormatArg::Hdf5, help_heading = "Output")]
    format: FormatArg,
    /// Write quiver overlay PNGs to <output>_quiver/.
    #[arg(long, help_heading = "Output")]
    quiver: bool,
    /// Also encode the quiver PNGs to <output>_quiver.mp4 (implies --quiver).
    #[arg(long, help_heading = "Output")]
    video: bool,
    /// Playback rate of the quiver video.
    #[arg(long, default_value_t = 30.0, help_heading = "Output")]
    video_fps: f64,
    /// Arrow length in image px per px of displacement (default: auto from the first pair).
    #[arg(long, help_heading = "Output")]
    quiver_scale: Option<f32>,
    /// Draw every n-th vector.
    #[arg(long, default_value_t = 1, help_heading = "Output")]
    quiver_every: usize,
    /// Plain quiver images: no vorticity / kinetic energy readout or chart.
    #[arg(long, help_heading = "Output")]
    no_overlay: bool,

    // ── Physical scale (optional) ───────────────────────────────────────────
    /// Measured scale in pixels per millimetre (preferred over FOV).
    #[arg(long, help_heading = "Scale")]
    px_per_mm: Option<f64>,
    /// Lens field of view in degrees (needs --sheet-distance).
    #[arg(long, help_heading = "Scale")]
    fov_deg: Option<f64>,
    /// Which frame dimension --fov-deg spans.
    #[arg(long, value_enum, default_value_t = AxisArg::Diagonal, help_heading = "Scale")]
    fov_axis: AxisArg,
    /// Camera-to-light-sheet distance in metres.
    #[arg(long, help_heading = "Scale")]
    sheet_distance: Option<f64>,
    /// Light-sheet thickness in mm (stride guidance; total kinetic energy of the slab).
    #[arg(long, help_heading = "Scale")]
    sheet_thickness_mm: Option<f64>,
    /// Fluid density in kg/m³ for kinetic energy (default: air at 20 °C).
    #[arg(long, default_value_t = AIR_DENSITY, help_heading = "Scale")]
    density: f64,
}

fn parse_background(s: &str) -> Result<Option<(BackgroundMode, usize)>> {
    if s == "none" {
        return Ok(None);
    }
    let (mode, n) = s
        .split_once(':')
        .with_context(|| format!("--background {s:?}: expected none, min:N or mean:N"))?;
    let mode = match mode {
        "min" => BackgroundMode::Min,
        "mean" => BackgroundMode::Mean,
        _ => bail!("--background {s:?}: mode must be min or mean"),
    };
    let n: usize = n
        .parse()
        .with_context(|| format!("--background {s:?}: bad frame count"))?;
    ensure!(n >= 1, "--background needs at least one frame");
    Ok(Some((mode, n)))
}

/// Frames held while their pairs are pending.
struct Held {
    prepared: PreparedFrame,
    raw: Option<Frame>,
}

#[derive(Default)]
struct Stats {
    pairs: usize,
    vectors: usize,
    no_signal: usize,
    rejected: usize,
    replaced: usize,
    magnitudes: Vec<f32>,
}

impl Stats {
    fn add(&mut self, f: &VectorField) {
        self.pairs += 1;
        for ((&fl, &u), &v) in f.flags.iter().zip(&f.dx).zip(&f.dy) {
            self.vectors += 1;
            if fl & flags::NO_SIGNAL != 0 {
                self.no_signal += 1;
            } else if fl & flags::INVALID != 0 {
                self.rejected += 1;
                if fl & flags::REPLACED != 0 {
                    self.replaced += 1;
                }
            } else if self.vectors % 3 == 0 {
                self.magnitudes.push(u.hypot(v));
            }
        }
    }
}

fn percentile(v: &mut [f32], q: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    *v.select_nth_unstable_by(k, f32::total_cmp).1
}

pub fn run(args: AnalyzeArgs) -> Result<()> {
    let t0 = Instant::now();
    let info = probe(&args.input)?;
    let capture = resolve_capture_rate(args.capture_fps, &info);
    if let Some(w) = &capture.warning {
        eprintln!("warning: {w}");
    }
    let rate = capture.rate;

    // Frame range.
    let to_frame = |s: f64| (s * rate.fps()).round().max(0.0) as usize;
    let start = args.start.or(args.start_s.map(to_frame)).unwrap_or(0);
    let end = args
        .end
        .or(args.end_s.map(to_frame))
        .unwrap_or(info.frame_count)
        .min(info.frame_count);
    ensure!(
        args.stride >= 1 && args.pair_step >= 1,
        "--stride and --pair-step must be ≥ 1"
    );
    ensure!(
        start + args.stride < end,
        "frame range [{start}, {end}) is too short for stride {} (clip has {} frames)",
        args.stride,
        info.frame_count
    );
    let pair_starts: Vec<usize> = (start..end - args.stride).step_by(args.pair_step).collect();
    let dt = rate.dt(args.stride as u64);

    // Output paths.
    let format = match args.format {
        FormatArg::Hdf5 => OutputFormat::Hdf5,
        FormatArg::Npz => OutputFormat::Npz,
    };
    if args.output.exists() && !args.force {
        bail!(
            "{} exists; pass --force to overwrite",
            args.output.display()
        );
    }
    if args.output.exists() {
        if args.output.is_dir() {
            std::fs::remove_dir_all(&args.output)?;
        } else {
            std::fs::remove_file(&args.output)?;
        }
    }
    let quiver = args.quiver || args.video;
    let quiver_dir = sibling(&args.output, "_quiver");
    if quiver {
        if quiver_dir.exists() {
            ensure!(
                args.force,
                "{} exists; pass --force to overwrite",
                quiver_dir.display()
            );
            std::fs::remove_dir_all(&quiver_dir)?;
        }
        std::fs::create_dir_all(&quiver_dir)?;
    }

    // Configuration.
    let channel = match args.channel {
        ChannelArg::Luma => Channel::Luma,
        ChannelArg::Green => Channel::Green,
        ChannelArg::Yplane => Channel::Yplane,
    };
    let decode = DecodeOptions {
        start,
        count: Some(end - start),
        channel,
        roi: args.roi,
        high_bit_depth: info.bit_depth > 8,
        container_rate: info.container_rate,
        frame_size: (info.height, info.width),
    };
    let preprocess = PreprocessConfig {
        smooth_sigma: args.smooth,
        normalize_radius: args.normalize_radius,
        ..Default::default()
    };
    let piv = PivConfig {
        windows: args.window.clone(),
        overlap: args.overlap,
        final_iterations: args.final_iterations,
        mask: MaskConfig {
            min_mean: args.min_mean,
            min_std: args.min_std,
        },
        validation: ValidationConfig {
            ppr_threshold: args.ppr,
            median_threshold: args.median_threshold,
            median_eps: args.median_eps,
            replace: !args.no_replace,
        },
    };
    let analyzer = PivAnalyzer::new(piv.clone())?;
    let scale_spec = ScaleSpec {
        px_per_mm: args.px_per_mm,
        fov_deg: args.fov_deg,
        fov_axis: Some(match args.fov_axis {
            AxisArg::Diagonal => FovAxis::Diagonal,
            AxisArg::Horizontal => FovAxis::Horizontal,
            AxisArg::Vertical => FovAxis::Vertical,
        }),
        sheet_distance_m: args.sheet_distance,
    };
    // The FOV spans the full frame, not the ROI.
    let scale = scale_spec.resolve(info.width, info.height);
    if args.fov_deg.is_some() && args.sheet_distance.is_none() && args.px_per_mm.is_none() {
        eprintln!(
            "note: --fov-deg without --sheet-distance gives no scale; output stays in pixel units"
        );
    }

    print_header(
        &info,
        &capture.detail,
        rate,
        start,
        end,
        args.stride,
        pair_starts.len(),
        dt,
    );

    // Background.
    let background_spec = parse_background(&args.background)?;
    let background = match background_spec {
        None => None,
        Some((mode, n)) => {
            let total = info.frame_count;
            let n = n.min(total);
            let mut est = BackgroundEstimator::new(mode);
            for k in 0..n {
                let idx = if n > 1 { k * (total - 1) / (n - 1) } else { 0 };
                est.add(&read_frame(&args.input, idx, &decode)?);
            }
            eprintln!("background: {mode:?} of {n} frames sampled across the clip");
            est.finish()
        }
    };

    // Decode on a separate thread so ffmpeg runs ahead of the analysis.
    let (tx, rx) = mpsc::sync_channel::<Result<(usize, Frame)>>(6);
    let input = args.input.clone();
    let decode_opts = decode.clone();
    let decoder = thread::spawn(move || {
        let mut src = match FfmpegSource::open(&input, &decode_opts) {
            Ok(s) => s,
            Err(e) => return tx.send(Err(e)).ok(),
        };
        loop {
            match src.next_frame() {
                Ok(Some(item)) => tx.send(Ok(item)).ok()?,
                Ok(None) => return Some(()),
                Err(e) => return tx.send(Err(e)).ok(),
            }
        }
    });

    let pngs = quiver.then(PngSink::spawn);

    let is_a = |f: usize| f >= start && f + args.stride < end && (f - start) % args.pair_step == 0;
    let is_b = |f: usize| f >= start + args.stride && is_a(f - args.stride);

    let mut held: BTreeMap<usize, Held> = BTreeMap::new();
    let mut writer: Option<Box<dyn FieldWriter>> = None;
    let mut style: Option<QuiverStyle> = None;
    let mut stats = Stats::default();
    let mut series = FlowSeries::default();
    let mut noise_px = Vec::new();
    let mut quiver_pngs = Vec::new();
    let mut grid_step = 0;
    let mut last_report = Instant::now();
    let show_progress = std::io::stderr().is_terminal();
    let t_loop = Instant::now();

    for item in rx {
        let (f, raw) = item?;
        let (a_role, b_role) = (is_a(f), is_b(f));
        if !a_role && !b_role {
            continue;
        }
        let prepared = PreparedFrame::new(&raw, &preprocess, background.as_ref());
        held.insert(
            f,
            Held {
                prepared,
                raw: (quiver && a_role).then_some(raw),
            },
        );

        if b_role {
            let a_idx = f - args.stride;
            let a = held.get(&a_idx).context("internal: frame A missing")?;
            let b = held.get(&f).expect("just inserted");
            let result = analyzer.analyze(&a.prepared, &b.prepared)?;
            let field = &result.field;

            let w = match &mut writer {
                Some(w) => w,
                None => {
                    let meta = RunMetadata {
                        tool_version: format!("piv {}", env!("CARGO_PKG_VERSION")),
                        source_path: info.path.clone(),
                        source_size: info.file_size,
                        camera: info.camera.clone(),
                        frame_width: info.width,
                        frame_height: info.height,
                        roi: args.roi,
                        channel,
                        capture_fps: rate.fps(),
                        capture_fps_rational: format!("{}/{}", rate.num(), rate.den()),
                        capture_fps_source: capture.source.clone(),
                        capture_fps_detail: capture.detail.clone(),
                        container_fps: info.container_rate.fps(),
                        start_frame: start,
                        end_frame: end,
                        stride: args.stride,
                        pair_step: args.pair_step,
                        dt_s: dt,
                        background: background_spec.map(|_| args.background.clone()),
                        preprocess: preprocess.clone(),
                        piv: piv.clone(),
                        scale_spec: scale_spec.clone(),
                        scale: scale.clone(),
                        sheet_thickness_mm: args.sheet_thickness_mm,
                    };
                    writer.insert(create_writer(format, &args.output, &meta, field)?)
                }
            };
            let time_s = rate.time_of(a_idx as f64 + args.stride as f64 / 2.0);
            w.write(&PairRecord {
                frame_a: a_idx,
                frame_b: f,
                time_s,
                field,
            })?;
            stats.add(field);
            let ok = usable(field.dx.view(), field.flags.view());
            series.push(
                time_s,
                pair_stats(field.dx.view(), field.dy.view(), &ok, field.grid.step, dt),
            );
            noise_px.push(neighbour_noise(&field.dx, &field.dy, &ok));
            grid_step = field.grid.step;

            if quiver {
                let raw_a = a.raw.as_ref().context("internal: raw frame missing")?;
                let s = *style.get_or_insert_with(|| {
                    let auto = QuiverStyle::auto(raw_a, field);
                    QuiverStyle {
                        arrow_scale: args.quiver_scale.unwrap_or(auto.arrow_scale),
                        every: args.quiver_every,
                        ..auto
                    }
                });
                let img = render_quiver(raw_a, field, &s);
                if let Some(p) = &pngs {
                    let path = quiver_dir.join(format!("quiver_f{a_idx:06}.png"));
                    quiver_pngs.push(path.clone());
                    p.send(path, img);
                }
            }
        }
        // Drop frames no later pair needs.
        let keep_from = (f + 1).saturating_sub(args.stride);
        held.retain(|&k, _| k >= keep_from);

        if show_progress && last_report.elapsed() > Duration::from_millis(500) {
            last_report = Instant::now();
            let rate = stats.pairs as f64 / t_loop.elapsed().as_secs_f64();
            eprint!(
                "\r  {}/{} pairs ({rate:.0} pairs/s)   ",
                stats.pairs,
                pair_starts.len()
            );
        }
    }
    if show_progress {
        eprint!("\r{:60}\r", "");
    }

    let decoded_ok = decoder
        .join()
        .map_err(|_| anyhow::anyhow!("decoder thread panicked"))?;
    ensure!(decoded_ok.is_some(), "decoding stopped early");
    let pngs = pngs.map(PngSink::finish).transpose()?.unwrap_or(0);
    match writer {
        Some(w) => w.finish()?,
        None => bail!("no frame pairs were analysed"),
    }
    let units = EnergyUnits {
        m_per_px: scale.as_ref().map(|s| s.m_per_px),
        thickness_m: args.sheet_thickness_mm.map(|t| t * 1e-3),
        density: args.density,
    };
    match format {
        #[cfg(feature = "hdf5")]
        OutputFormat::Hdf5 => {
            piv_io::flow::write_stats_hdf5(&args.output, "stats", &series, &units)?
        }
        #[cfg(not(feature = "hdf5"))]
        OutputFormat::Hdf5 => unreachable!("create_writer refuses HDF5 without the feature"),
        OutputFormat::Npz => piv_io::flow::write_stats_npz(&args.output, &series)?,
    }
    ensure!(
        stats.pairs == pair_starts.len(),
        "analysed {} pairs, expected {} — the clip may be shorter than reported",
        stats.pairs,
        pair_starts.len()
    );

    let elapsed = t_loop.elapsed().as_secs_f64();
    println!(
        "analysed {} pairs in {elapsed:.1} s ({:.1} pairs/s) → {}",
        stats.pairs,
        stats.pairs as f64 / elapsed,
        args.output.display()
    );
    if quiver && !args.no_overlay {
        let t = Instant::now();
        let chart = run_chart(&[&series], &units);
        let n = series.len();
        quiver_pngs
            .par_iter()
            .enumerate()
            .try_for_each(|(i, path)| {
                let lines = readout(None, series.t[i], i, n, &series.stats[i], &units);
                decorate_png(path, &lines, &chart, i)
            })?;
        eprintln!("overlay: {:.1} s", t.elapsed().as_secs_f64());
    }
    if quiver {
        println!("quiver: {pngs} PNGs in {}", quiver_dir.display());
    }
    if args.video {
        let video = sibling(&args.output, "_quiver.mp4");
        encode_video(&quiver_dir, &video, args.video_fps)?;
        println!("video:  {}", video.display());
    }
    print_summary(
        &mut stats,
        dt,
        analyzer.config(),
        args.stride,
        scale.as_ref().map(|s| s.m_per_px),
        args.sheet_thickness_mm,
        &args,
    );
    print_flow(&series, &mut noise_px, dt, grid_step, &units);
    eprintln!("total time {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Run-wide vorticity and kinetic energy, and the noise floor of the vorticity.
fn print_flow(
    series: &FlowSeries,
    noise_px: &mut [f32],
    dt: f64,
    step: usize,
    units: &EnergyUnits,
) {
    let col = |f: fn(&piv_core::FlowStats) -> f32| -> Vec<f32> {
        series
            .column(f)
            .into_iter()
            .filter(|x| x.is_finite())
            .collect()
    };
    let mut w = col(|s| s.vorticity_mean);
    if w.is_empty() {
        return;
    }
    let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
    let mut wa = col(|s| s.vorticity_abs_mean);
    let mut e: Vec<f32> = col(|s| s.ke_sum)
        .into_iter()
        .map(|x| units.ke_total(x).0 as f32)
        .collect();
    let (_, ke_unit) = units.ke_total(0.0);
    println!(
        "flow:    ω mean {:+.2} 1/s over the run (pairs {:+.2} … {:+.2}); |ω| mean {:.2} 1/s",
        mean(&w),
        percentile(&mut w, 0.0),
        percentile(&mut w, 1.0),
        mean(&wa),
    );
    println!(
        "         KE median {} {ke_unit}, peak {} {ke_unit} ({})",
        piv_io::overlay::fmt_num(percentile(&mut e, 0.5) as f64),
        piv_io::overlay::fmt_num(percentile(&mut e, 1.0) as f64),
        units.ke_total_label()
    );
    // Per-point ω noise with central differences is σ_u / step; |ω| of pure noise averages
    // √(2/π) of that.
    let mut noise: Vec<f32> = noise_px.iter().copied().filter(|x| x.is_finite()).collect();
    let sigma_px = percentile(&mut noise, 0.5);
    if sigma_px.is_finite() {
        let sigma_w = sigma_px as f64 / dt / step.max(1) as f64;
        println!(
            "noise:   ≈{sigma_px:.2} px per vector (neighbour residual, an upper bound) → ω ±{sigma_w:.1} 1/s per point; \
             |ω| mean of noise alone ≈ {:.1} 1/s (median |ω| mean here {:.1})",
            sigma_w * (2.0 / std::f64::consts::PI).sqrt(),
            percentile(&mut wa, 0.5)
        );
    }
}

/// `out/run.h5` + `_quiver` → `out/run_quiver`.
fn sibling(output: &Path, suffix: &str) -> PathBuf {
    let stem = output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "piv".into());
    output.with_file_name(format!("{stem}{suffix}"))
}

#[allow(clippy::too_many_arguments)]
fn print_header(
    info: &MediaInfo,
    rate_detail: &str,
    rate: FrameRate,
    start: usize,
    end: usize,
    stride: usize,
    pairs: usize,
    dt: f64,
) {
    println!(
        "input:   {} — {}×{}, {} frames",
        info.path, info.width, info.height, info.frame_count
    );
    println!(
        "timing:  capture {rate} [{rate_detail}]; stride {stride} → dt = {:.4} ms",
        dt * 1e3
    );
    println!("range:   frames [{start}, {end}) → {pairs} pairs");
}

#[allow(clippy::too_many_arguments)]
fn print_summary(
    stats: &mut Stats,
    dt: f64,
    cfg: &PivConfig,
    stride: usize,
    m_per_px: Option<f64>,
    sheet_mm: Option<f64>,
    args: &AnalyzeArgs,
) {
    let pct = |n: usize| 100.0 * n as f64 / stats.vectors.max(1) as f64;
    let valid = stats.vectors - stats.no_signal - stats.rejected;
    println!(
        "vectors: {:.1}% valid, {:.1}% no signal, {:.1}% rejected ({:.1}% replaced)",
        pct(valid),
        pct(stats.no_signal),
        pct(stats.rejected),
        pct(stats.replaced)
    );
    let m = &mut stats.magnitudes;
    let (p50, p95) = (percentile(m, 0.5), percentile(m, 0.95));
    if !p50.is_finite() {
        println!("no valid vectors — check --min-mean/--min-std, --roi and the frame range");
        return;
    }
    let to_v = |d: f32| d as f64 / dt;
    print!(
        "|disp|:  median {p50:.2} px, p95 {p95:.2} px  →  |v| median {:.0} px/s, p95 {:.0} px/s",
        to_v(p50),
        to_v(p95)
    );
    match m_per_px {
        Some(k) => println!(" ≈ {:.3} m/s, {:.3} m/s", to_v(p50) * k, to_v(p95) * k),
        None => println!(),
    }

    // Stride guidance. Much of a fog field is nearly static, so judge the stride by the
    // faster-moving vectors (90th percentile) rather than the median.
    let p90 = percentile(m, 0.9);
    let window = *cfg.windows.last().expect("validated") as f32;
    let signal = stats.vectors - stats.no_signal;
    let rejected_frac = stats.rejected as f32 / signal.max(1) as f32;
    println!(
        "         90th percentile {p90:.2} px; {:.0}% of seeded vectors rejected",
        100.0 * rejected_frac
    );
    if p90 < 2.0 {
        let suggest = ((4.0 / p90.max(0.1) * stride as f32).round() as usize).max(stride + 1);
        println!(
            "hint:    most motion is under 2 px per pair, so relative error is high — try --stride {suggest}"
        );
    } else if p90 > window / 4.0 {
        let suggest = ((stride as f32 * window / 8.0 / p90).floor() as usize).max(1);
        if suggest < stride {
            println!(
                "hint:    fast regions move more than 1/4 window per pair — try --stride {suggest}"
            );
        } else {
            println!(
                "hint:    fast regions move more than 1/4 window per pair — try larger windows"
            );
        }
    } else if rejected_frac > 0.3 {
        println!("hint:    many vectors rejected — try larger windows or a smaller --stride");
    }
    if let Some(t) = sheet_mm {
        println!(
            "sheet:   {t} mm thick → out-of-plane speeds above ~{:.2} m/s lose particles between frames at this dt",
            t * 1e-3 / 4.0 / dt
        );
    }
    if args.fov_deg.is_some() && m_per_px.is_some() && args.px_per_mm.is_none() {
        println!(
            "scale:   FOV-based estimate assumes the full sensor; high-frame-rate modes that crop the sensor make \
             m/s values an upper bound"
        );
    }
}
