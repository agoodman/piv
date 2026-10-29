use std::path::Path;

use anyhow::Result;
use piv_core::FrameRate;
use piv_io::{probe, resolve_capture_rate};

pub fn run(input: &Path, capture_fps: Option<FrameRate>, json: bool) -> Result<()> {
    let info = probe(input)?;
    let capture = resolve_capture_rate(capture_fps, &info);
    if json {
        let v = serde_json::json!({ "media": info, "capture": capture });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    let mb = info.file_size as f64 / 1e6;
    println!("file:           {} ({mb:.1} MB)", info.path);
    if let Some(c) = &info.camera {
        println!("camera:         {c}");
    }
    let transfer = info.color_transfer.as_deref().unwrap_or("unknown");
    println!(
        "video:          {} {}×{} {} ({}-bit), transfer {transfer}",
        info.codec, info.width, info.height, info.pix_fmt, info.bit_depth
    );
    println!("frames:         {}", info.frame_count);
    println!(
        "container rate: {} ({:.3} s playback)",
        info.container_rate,
        info.playback_duration()
    );
    if let Some(c) = &info.camera_rate {
        println!("camera rate:    {} ({}: {:?})", c.rate, c.source, c.raw);
    }
    let real = capture.rate.time_of(info.frame_count as f64);
    println!("capture rate:   {} [{}]", capture.rate, capture.detail);
    println!(
        "real duration:  {real:.4} s, dt = {:.4} ms per frame",
        capture.rate.dt(1) * 1e3
    );
    let slow = capture.rate.fps() / info.container_rate.fps();
    if (slow - 1.0).abs() > 1e-6 {
        println!("slow motion:    {slow:.3}× (playback is slowed relative to capture)");
    }
    if let Some(w) = &capture.warning {
        eprintln!("warning: {w}");
    }
    Ok(())
}
