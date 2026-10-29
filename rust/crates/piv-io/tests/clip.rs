//! Tests against the real sample clip. Ignored by default because the clip is not in git:
//! `cargo test -p piv-io -- --ignored`.

use std::path::PathBuf;

use piv_core::FrameRate;
use piv_io::decode::{DecodeOptions, FfmpegSource, FrameSource, read_frame};
use piv_io::{Channel, probe, resolve_capture_rate};

fn clip() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../clips/C0141.MP4")
}

fn opts(info: &piv_io::MediaInfo) -> DecodeOptions {
    DecodeOptions {
        start: 0,
        count: None,
        channel: Channel::Luma,
        roi: None,
        high_bit_depth: false,
        container_rate: info.container_rate,
        frame_size: (info.height, info.width),
    }
}

#[test]
#[ignore = "needs clips/C0141.MP4"]
fn probe_c0141() {
    let info = probe(&clip()).unwrap();
    assert_eq!(
        (info.width, info.height, info.frame_count, info.bit_depth),
        (1920, 1080, 3720, 8)
    );
    assert_eq!(info.container_rate, FrameRate::new(50, 1).unwrap());
    assert_eq!(info.camera.as_deref(), Some("Sony DSC-RX100M5A"));
    let r = resolve_capture_rate(None, &info);
    assert_eq!(r.rate, FrameRate::new(1000, 1).unwrap());
    assert!(r.warning.is_none());
}

#[test]
#[ignore = "needs clips/C0141.MP4"]
fn seeking_is_frame_accurate() {
    let info = probe(&clip()).unwrap();
    let o = opts(&info);
    // Decode frames 245..255 sequentially, then seek straight to some of them.
    let mut src = FfmpegSource::open(
        &clip(),
        &DecodeOptions {
            start: 245,
            count: Some(10),
            ..o.clone()
        },
    )
    .unwrap();
    let mut seq = Vec::new();
    while let Some((i, f)) = src.next_frame().unwrap() {
        seq.push((i, f));
    }
    assert_eq!(seq.len(), 10);
    assert_eq!(seq[0].0, 245);
    for (i, f) in seq.iter().step_by(3) {
        let direct = read_frame(&clip(), *i, &o).unwrap();
        let diff = (&direct - f).mapv(f32::abs).sum();
        assert_eq!(diff, 0.0, "frame {i} differs when seeked");
    }
    // Consecutive frames must differ (no duplicated HFR frames).
    let d = (&seq[1].1 - &seq[0].1).mapv(f32::abs).mean().unwrap();
    assert!(d > 0.001, "{d}");
}

#[test]
#[ignore = "needs clips/C0141.MP4"]
fn roi_and_green_channel() {
    let info = probe(&clip()).unwrap();
    let o = DecodeOptions {
        roi: Some("1200,400,640,480".parse().unwrap()),
        channel: Channel::Green,
        ..opts(&info)
    };
    let f = read_frame(&clip(), 250, &o).unwrap();
    assert_eq!(f.dim(), (480, 640));
    assert!(f.mean().unwrap() > 0.0);
}
