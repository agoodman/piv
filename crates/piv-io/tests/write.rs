use piv_core::{FrameRate, Grid, PivConfig, PreprocessConfig, VectorField, flags};
use piv_io::decode::{Channel, Roi};
use piv_io::metadata::RateSource;
use piv_io::scale::{FovAxis, ScaleSpec};
use piv_io::write::{OutputFormat, PairRecord, RunMetadata, create_writer};

fn meta(scale: bool) -> RunMetadata {
    let rate = FrameRate::new(1000, 1).unwrap();
    let scale_spec = ScaleSpec {
        fov_deg: scale.then_some(84.0),
        fov_axis: Some(FovAxis::Diagonal),
        sheet_distance_m: scale.then_some(0.3),
        ..Default::default()
    };
    RunMetadata {
        tool_version: "test".into(),
        source_path: "clip.mp4".into(),
        source_size: 1,
        camera: Some("Sony DSC-RX100M5A".into()),
        frame_width: 1920,
        frame_height: 1080,
        roi: Some(Roi {
            x: 100,
            y: 50,
            width: 128,
            height: 96,
        }),
        channel: Channel::Luma,
        capture_fps: rate.fps(),
        capture_fps_rational: "1000/1".into(),
        capture_fps_source: RateSource::CameraMetadata,
        capture_fps_detail: "Sony NonRealTimeMeta (1000p)".into(),
        container_fps: 50.0,
        start_frame: 10,
        end_frame: 20,
        stride: 2,
        pair_step: 1,
        dt_s: rate.dt(2),
        background: None,
        preprocess: PreprocessConfig::default(),
        piv: PivConfig::default(),
        scale: scale_spec.resolve(1920, 1080),
        scale_spec,
        sheet_thickness_mm: Some(1.0),
    }
}

fn field(k: f32) -> VectorField {
    let mut f = VectorField::new(Grid::new(96, 128, 32, 0.5).unwrap());
    f.dx.fill(1.5 * k);
    f.dy.fill(-0.5 * k);
    f.peak_ratio.fill(3.0);
    f.flags.fill(0);
    f.flags[[0, 0]] = flags::NO_SIGNAL;
    f.dx[[0, 0]] = f32::NAN;
    f
}

#[test]
fn hdf5_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.h5");
    let m = meta(true);
    let fields = [field(1.0), field(2.0), field(3.0)];
    let mut w = create_writer(OutputFormat::Hdf5, &path, &m, &fields[0]).unwrap();
    for (i, f) in fields.iter().enumerate() {
        w.write(&PairRecord {
            frame_a: 10 + i,
            frame_b: 12 + i,
            time_s: (11 + i) as f64 * 1e-3,
            field: f,
        })
        .unwrap();
    }
    w.finish().unwrap();

    let file = hdf5_metno::File::open(&path).unwrap();
    let (ny, nx) = fields[0].dx.dim();
    let dx = file.dataset("fields/dx").unwrap();
    assert_eq!(dx.shape(), vec![3, ny, nx]);
    let dx: ndarray::Array3<f32> = dx.read().unwrap();
    assert_eq!(dx[[2, 1, 1]], 4.5);
    assert!(dx[[0, 0, 0]].is_nan());

    let u: ndarray::Array3<f32> = file.dataset("fields/u").unwrap().read().unwrap();
    assert!(
        (u[[0, 1, 1]] - 750.0).abs() < 1e-3,
        "1.5 px over 2 ms = 750 px/s"
    );
    let u_mps: ndarray::Array3<f32> = file.dataset("fields/u_mps").unwrap().read().unwrap();
    assert!((u_mps[[0, 1, 1]] - 750.0 * 0.000_245_240_7).abs() < 1e-4);

    let flags: ndarray::Array3<u8> = file.dataset("fields/flags").unwrap().read().unwrap();
    assert_eq!(flags[[1, 0, 0]], flags::NO_SIGNAL);

    let x: Vec<f32> = file.dataset("grid/x").unwrap().read_raw().unwrap();
    assert_eq!(x[0], 100.0 + 15.5, "ROI offset is added back");
    let fa: Vec<u64> = file.dataset("pairs/frame_a").unwrap().read_raw().unwrap();
    assert_eq!(fa, vec![10, 11, 12]);
    let t: Vec<f64> = file.dataset("pairs/t").unwrap().read_raw().unwrap();
    assert!((t[2] - 0.013).abs() < 1e-12);

    let fps: f64 = file.attr("capture_fps").unwrap().read_scalar().unwrap();
    assert_eq!(fps, 1000.0);
    let stride: u64 = file.attr("stride").unwrap().read_scalar().unwrap();
    assert_eq!(stride, 2);
    let src: hdf5_metno::types::VarLenUnicode = file
        .attr("capture_fps_source")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(src.as_str(), "camera_metadata");
    let json: hdf5_metno::types::VarLenUnicode =
        file.attr("run_json").unwrap().read_scalar().unwrap();
    let v: serde_json::Value = serde_json::from_str(json.as_str()).unwrap();
    assert_eq!(v["piv"]["windows"], serde_json::json!([96, 64]));
}

#[test]
fn hdf5_without_scale_has_no_physical_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.h5");
    let f = field(1.0);
    let mut w = create_writer(OutputFormat::Hdf5, &path, &meta(false), &f).unwrap();
    w.write(&PairRecord {
        frame_a: 0,
        frame_b: 2,
        time_s: 0.001,
        field: &f,
    })
    .unwrap();
    w.finish().unwrap();
    let file = hdf5_metno::File::open(&path).unwrap();
    assert!(file.dataset("fields/u_mps").is_err());
    assert!(file.dataset("fields/u").is_ok());
}

#[test]
fn npz_writes_pairs_and_run_json() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("npz");
    let f = field(1.0);
    let mut w = create_writer(OutputFormat::Npz, &out, &meta(true), &f).unwrap();
    w.write(&PairRecord {
        frame_a: 10,
        frame_b: 12,
        time_s: 0.011,
        field: &f,
    })
    .unwrap();
    w.finish().unwrap();
    assert!(out.join("run.json").exists());
    let mut npz = ndarray_npy::NpzReader::new(
        std::fs::File::open(out.join("pair_000010_000012.npz")).unwrap(),
    )
    .unwrap();
    let names = npz.names().unwrap();
    for n in [
        "x",
        "y",
        "dx",
        "dy",
        "u",
        "v",
        "u_mps",
        "v_mps",
        "peak_ratio",
        "flags",
        "frames",
        "t",
    ] {
        assert!(
            names.iter().any(|m| m.trim_end_matches(".npy") == n),
            "missing {n}: {names:?}"
        );
    }
    let dx: ndarray::Array2<f32> = npz.by_name("dx").unwrap();
    assert_eq!(dx[[1, 1]], 1.5);
}
