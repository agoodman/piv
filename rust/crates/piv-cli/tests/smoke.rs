//! End-to-end run of the `piv` binary on the sample clip. Ignored by default because the clip
//! is not in git: `cargo test -p piv-cli -- --ignored`.

use std::path::PathBuf;
use std::process::Command;

fn clip() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../clips/C0141.MP4")
}

#[test]
#[ignore = "needs clips/C0141.MP4"]
fn info_reports_capture_rate() {
    let out = Command::new(env!("CARGO_BIN_EXE_piv"))
        .arg("info")
        .arg(clip())
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("capture rate:   1000 fps"), "{text}");
    assert!(text.contains("Sony DSC-RX100M5A"), "{text}");
}

#[test]
#[ignore = "needs clips/C0141.MP4"]
fn analyze_writes_hdf5_quiver_and_video() {
    let dir = tempfile::tempdir().unwrap();
    let h5 = dir.path().join("smoke.h5");
    let out = Command::new(env!("CARGO_BIN_EXE_piv"))
        .arg("analyze")
        .arg(clip())
        .arg("-o")
        .arg(&h5)
        .args([
            "--start", "200", "--end", "260", "--stride", "5", "--quiver", "--video",
        ])
        .args([
            "--fov-deg",
            "84",
            "--sheet-distance",
            "0.3",
            "--sheet-thickness-mm",
            "1",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("analysed 55 pairs"), "{stdout}");
    assert!(stdout.contains("flow:    ω mean"), "{stdout}");
    assert!(stdout.contains("µJ"), "{stdout}");

    let file = hdf5_metno::File::open(&h5).unwrap();
    let nx = file.dataset("grid/x").unwrap().shape()[0];
    let ny = file.dataset("grid/y").unwrap().shape()[0];
    // Default final window 64 px at 50 % overlap on 1920×1080.
    assert_eq!((nx, ny), (59, 32));
    assert_eq!(file.dataset("fields/u").unwrap().shape(), vec![55, ny, nx]);
    assert!(file.dataset("fields/u_mps").is_ok());
    let dt: f64 = file.attr("dt_s").unwrap().read_scalar().unwrap();
    assert!((dt - 0.005).abs() < 1e-12);

    let flags: ndarray::Array3<u8> = file.dataset("fields/flags").unwrap().read().unwrap();
    let dx: ndarray::Array3<f32> = file.dataset("fields/dx").unwrap().read().unwrap();
    let dy: ndarray::Array3<f32> = file.dataset("fields/dy").unwrap().read().unwrap();
    let seeded = flags.iter().filter(|&&f| f & 1 == 0).count();
    let valid = flags.iter().filter(|&&f| f & 23 == 0).count();
    // This fog footage is hard (thin, low-texture regions, out-of-plane motion): ~45 % of
    // seeded vectors pass validation here. Guard against regressions well below that.
    assert!(
        seeded > 0 && valid * 10 > seeded * 3,
        "valid {valid} of {seeded} seeded vectors"
    );
    // Valid displacements stay within what the 96 px first pass can measure, and typical
    // motion over 5 ms is a few px (the plume falls at up to ~3 px/ms early in the clip).
    let mut mags: Vec<f32> = flags
        .iter()
        .zip(&dx)
        .zip(&dy)
        .filter(|((f, _), _)| *f & 23 == 0)
        .map(|((_, u), v)| u.hypot(*v))
        .collect();
    mags.sort_by(f32::total_cmp);
    let (median, max) = (mags[mags.len() / 2], *mags.last().unwrap());
    assert!(max <= 48.0, "max |d| {max}");
    assert!((0.1..10.0).contains(&median), "median |d| {median}");

    for name in [
        "vorticity_mean",
        "vorticity_abs_mean",
        "ke_mean",
        "ke_sum",
        "coverage",
    ] {
        let ds = file.dataset(&format!("stats/{name}")).unwrap();
        assert_eq!(ds.shape(), vec![55], "{name}");
    }
    let cov: Vec<f32> = file.dataset("stats/coverage").unwrap().read_raw().unwrap();
    assert!(cov.iter().all(|c| (0.0..=1.0).contains(c)));

    // Overlays sit on top of full-width frames without changing their size.
    let png = std::fs::read_dir(dir.path().join("smoke_quiver"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert_eq!(png_header(&png), (1920, 1080));

    let pngs = std::fs::read_dir(dir.path().join("smoke_quiver"))
        .unwrap()
        .count();
    assert_eq!(pngs, 55);
    assert!(
        dir.path()
            .join("smoke_quiver.mp4")
            .metadata()
            .unwrap()
            .len()
            > 10_000
    );
}

#[test]
#[ignore = "needs clips/C0141.MP4"]
fn smooth_sweep_and_output() {
    let dir = tempfile::tempdir().unwrap();
    let h5 = dir.path().join("run.h5");
    let piv = env!("CARGO_BIN_EXE_piv");
    let ok = Command::new(piv)
        .arg("analyze")
        .arg(clip())
        .arg("-o")
        .arg(&h5)
        .args(["--start", "1400", "--end", "1500", "--stride", "5"])
        .status()
        .unwrap();
    assert!(ok.success());

    let out = Command::new(piv)
        .arg("smooth")
        .arg(&h5)
        .arg("--sweep")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}");
    assert!(
        text.contains("temporal median 9") && text.contains("lowest CV error"),
        "{text}"
    );

    let sm = dir.path().join("smooth.h5");
    let out = Command::new(piv)
        .arg("smooth")
        .arg(&h5)
        .arg("-o")
        .arg(&sm)
        .args(["--temporal", "9", "--spatial", "1", "--video"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let file = hdf5_metno::File::open(&sm).unwrap();
    let u: ndarray::Array3<f32> = file.dataset("fields/u").unwrap().read().unwrap();
    let raw: ndarray::Array3<f32> = file.dataset("raw/dx").unwrap().read().unwrap();
    for g in ["stats", "raw/stats"] {
        let n = file.dataset(&format!("{g}/ke_sum")).unwrap().shape()[0];
        assert_eq!(n, raw.shape()[0], "{g}");
    }
    assert_eq!(u.dim(), raw.dim());
    // The spatial pass never fills points the temporal pass rejected (default min support 3).
    let support: ndarray::Array3<u16> = file.dataset("fields/support").unwrap().read().unwrap();
    assert!(
        u.iter()
            .zip(&support)
            .all(|(u, &s)| !u.is_finite() || s >= 3)
    );
    assert!(
        dir.path()
            .join("smooth_compare.mp4")
            .metadata()
            .unwrap()
            .len()
            > 10_000
    );

    // Even window sizes are rejected.
    let bad = Command::new(piv)
        .arg("smooth")
        .arg(&h5)
        .arg("-o")
        .arg(dir.path().join("x.h5"))
        .args(["--temporal", "4"])
        .output()
        .unwrap();
    assert!(!bad.status.success());

    // A min support larger than the window would give an empty output.
    let bad = Command::new(piv)
        .arg("smooth")
        .arg(&h5)
        .arg("-o")
        .arg(dir.path().join("x.h5"))
        .args(["--temporal", "5", "--min-support", "9"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(!dir.path().join("x.h5").exists());

    // An existing comparison directory is reported before anything is written.
    std::fs::create_dir(dir.path().join("y_compare")).unwrap();
    let bad = Command::new(piv)
        .arg("smooth")
        .arg(&h5)
        .arg("-o")
        .arg(dir.path().join("y.h5"))
        .args(["--temporal", "5", "--video"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(!dir.path().join("y.h5").exists());
}

/// Width and height from a PNG's IHDR chunk.
fn png_header(path: &std::path::Path) -> (u32, u32) {
    let b = std::fs::read(path).unwrap();
    let be = |i: usize| u32::from_be_bytes(b[i..i + 4].try_into().unwrap());
    (be(16), be(20))
}
