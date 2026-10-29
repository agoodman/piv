//! Writes reference data for the Swift/Metal port (PIVKit) to compare against, stage by stage.
//!
//! ```sh
//! cargo run -p piv-core --example fixtures --release            # default output dir
//! cargo run -p piv-core --example fixtures --release -- <dir>
//! ```
//!
//! Each case is a directory holding `meta.json` and raw little-endian arrays (`*.f32`, `*.u8`),
//! row-major, named in the JSON. The default output is
//! `swift/Tests/PIVKitTests/Fixtures/cases`.

use std::fs;
use std::path::{Path, PathBuf};

use piv_core::{
    Frame, MaskConfig, PivAnalyzer, PivConfig, PreparedFrame, PreprocessConfig, VectorField,
    synthetic,
};
use serde_json::{Value, json};

struct Case {
    name: &'static str,
    a: Frame,
    b: Frame,
    background: Option<Frame>,
    pre: PreprocessConfig,
    piv: PivConfig,
}

const RAW: PreprocessConfig = PreprocessConfig {
    smooth_sigma: 0.0,
    normalize_radius: 0,
    normalize_eps: 0.0,
};

fn piv(windows: &[usize]) -> PivConfig {
    PivConfig {
        windows: windows.to_vec(),
        final_iterations: if windows.len() == 1 { 1 } else { 2 },
        ..Default::default()
    }
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();

    let (a, b) = synthetic::particle_pair(128, 128, 12, 0.04, 1.2, |_, _| (3.7, -1.3));
    v.push(Case { name: "particles_uniform", a, b, background: None, pre: RAW, piv: piv(&[64, 32]) });

    let (c, om) = (95.5, 0.03);
    let (a, b) = synthetic::particle_pair(192, 192, 6, 0.04, 1.2, move |x, y| {
        (-om * (y - c), om * (x - c))
    });
    v.push(Case { name: "particles_rotation", a, b, background: None, pre: RAW, piv: piv(&[64, 32]) });

    // Radix-3 sizes in the fused path: 48 = 2⁴·3, 24 = 2³·3.
    let (c, g) = (79.5, 0.025);
    let (a, b) = synthetic::particle_pair(160, 160, 8, 0.04, 1.2, move |_, y| (g * (y - c), 0.0));
    v.push(Case { name: "particles_shear_48_24", a, b, background: None, pre: RAW, piv: piv(&[48, 24]) });

    let (a, b) = synthetic::particle_pair(256, 256, 5, 0.04, 1.2, |_, _| (20.0, -6.0));
    v.push(Case { name: "particles_large_shift", a, b, background: None, pre: RAW, piv: piv(&[64, 32]) });

    // The default 96 → 64 windows (batched 96 px path) on fog texture with the default
    // preprocessing (blur + local normalization).
    let (mut a, mut b) = synthetic::fog_pair(192, 192, 4, |_, _| (2.4, -1.7));
    synthetic::add_noise(&mut a, 0.01, 1);
    synthetic::add_noise(&mut b, 0.01, 2);
    v.push(Case {
        name: "fog_default",
        a,
        b,
        background: None,
        pre: PreprocessConfig::default(),
        piv: PivConfig::default(),
    });

    // Background subtraction and the window signal mask (the CLI's default thresholds).
    let (mut a, mut b) = synthetic::fog_pair(160, 160, 7, |x, _| (1.0 + 0.01 * x, 0.5));
    let bg = Frame::from_shape_fn((160, 160), |(y, x)| 0.05 + 0.0005 * (x + y) as f32);
    a += &bg;
    b += &bg;
    synthetic::add_noise(&mut a, 0.005, 3);
    synthetic::add_noise(&mut b, 0.005, 4);
    v.push(Case {
        name: "fog_background_mask",
        a,
        b,
        background: Some(bg),
        pre: PreprocessConfig::default(),
        piv: PivConfig {
            mask: MaskConfig { min_mean: 0.04, min_std: 0.015 },
            ..piv(&[64, 32])
        },
    });

    // Left half dark except faint noise: NO_SIGNAL there.
    let (mut a, mut b) = synthetic::particle_pair(128, 128, 9, 0.04, 1.2, |_, _| (1.0, 0.0));
    for img in [&mut a, &mut b] {
        img.slice_mut(ndarray::s![.., ..64]).fill(0.0);
    }
    synthetic::add_noise(&mut a, 0.002, 3);
    synthetic::add_noise(&mut b, 0.002, 4);
    v.push(Case {
        name: "particles_blank_half",
        a,
        b,
        background: None,
        pre: RAW,
        piv: PivConfig {
            mask: MaskConfig { min_mean: 0.01, min_std: 0.01 },
            ..Default::default()
        },
    });
    v
}

fn write_f32(dir: &Path, name: &str, data: impl IntoIterator<Item = f32>) -> String {
    let bytes: Vec<u8> = data.into_iter().flat_map(f32::to_le_bytes).collect();
    let file = format!("{name}.f32");
    fs::write(dir.join(&file), bytes).unwrap();
    file
}

fn write_u8(dir: &Path, name: &str, data: impl IntoIterator<Item = u8>) -> String {
    let file = format!("{name}.u8");
    fs::write(dir.join(&file), data.into_iter().collect::<Vec<u8>>()).unwrap();
    file
}

fn write_field(dir: &Path, prefix: &str, f: &VectorField) -> Value {
    json!({
        "dx": write_f32(dir, &format!("{prefix}_dx"), f.dx.iter().copied()),
        "dy": write_f32(dir, &format!("{prefix}_dy"), f.dy.iter().copied()),
        "peak_ratio": write_f32(dir, &format!("{prefix}_ppr"), f.peak_ratio.iter().copied()),
        "flags": write_u8(dir, &format!("{prefix}_flags"), f.flags.iter().copied()),
    })
}

fn main() {
    let out = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../swift/Tests/PIVKitTests/Fixtures/cases")
    });
    let mut names = Vec::new();
    for case in cases() {
        let dir = out.join(case.name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let (h, w) = case.a.dim();

        let pa = PreparedFrame::new(&case.a, &case.pre, case.background.as_ref());
        let pb = PreparedFrame::new(&case.b, &case.pre, case.background.as_ref());
        let analyzer = PivAnalyzer::new(case.piv.clone()).unwrap();
        let (result, traces) = analyzer.analyze_traced(&pa, &pb).unwrap();

        let passes: Vec<Value> = traces
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let g = &t.validated.grid;
                json!({
                    "grid": { "window": g.window, "step": g.step, "nx": g.nx, "ny": g.ny, "x0": g.x0, "y0": g.y0 },
                    "origins": write_f32(&dir, &format!("pass{i}_origins"), t.origins.iter().flatten().copied()),
                    "raw": write_field(&dir, &format!("pass{i}_raw"), &t.raw),
                    "validated": write_field(&dir, &format!("pass{i}"), &t.validated),
                })
            })
            .collect();

        let meta = json!({
            "name": case.name,
            "height": h,
            "width": w,
            "frame_a": write_f32(&dir, "frame_a", case.a.iter().copied()),
            "frame_b": write_f32(&dir, "frame_b", case.b.iter().copied()),
            "background": case.background.as_ref().map(|bg| write_f32(&dir, "background", bg.iter().copied())),
            "prepared_a": write_f32(&dir, "prepared_a", pa.image.iter().copied()),
            "prepared_b": write_f32(&dir, "prepared_b", pb.image.iter().copied()),
            "preprocess": case.pre,
            "piv": case.piv,
            "passes": passes,
            "summary": {
                "no_signal": result.summary.no_signal,
                "low_ppr": result.summary.low_ppr,
                "median_outliers": result.summary.median_outliers,
                "out_of_range": result.summary.out_of_range,
                "replaced": result.summary.replaced,
            },
        });
        fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
        println!(
            "{:<24} {w}×{h}  {} passes  {} vectors  {:?}",
            case.name,
            traces.len(),
            result.field.grid.len(),
            result.summary
        );
        names.push(case.name);
    }
    fs::write(out.join("index.json"), serde_json::to_string_pretty(&names).unwrap()).unwrap();
}
