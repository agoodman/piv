//! Media probing and capture-rate discovery.
//!
//! High-speed cameras often record at one rate and write the file at a slower playback rate
//! (C0141.MP4: captured at 1000 fps, stored as 50p). Velocities must use the capture rate, so
//! it is resolved from, in order: a user override, camera-specific metadata, the container.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use piv_core::FrameRate;
use serde::Serialize;
use serde_json::Value;

/// What ffprobe and the camera probes found out about a video file.
#[derive(Debug, Clone, Serialize)]
pub struct MediaInfo {
    pub path: String,
    pub file_size: u64,
    pub codec: String,
    pub width: usize,
    pub height: usize,
    pub pix_fmt: String,
    pub bit_depth: u32,
    pub color_transfer: Option<String>,
    pub frame_count: usize,
    pub container_rate: FrameRate,
    pub camera: Option<String>,
    pub camera_rate: Option<CameraRate>,
}

/// A capture rate read from camera-specific metadata.
#[derive(Debug, Clone, Serialize)]
pub struct CameraRate {
    pub rate: FrameRate,
    /// Which probe found it, e.g. "Sony NonRealTimeMeta".
    pub source: String,
    /// The raw value as written by the camera, e.g. "1000p".
    pub raw: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RateSource {
    Override,
    CameraMetadata,
    Container,
}

/// The rate used for timing, and where it came from.
#[derive(Debug, Clone, Serialize)]
pub struct CaptureRate {
    pub rate: FrameRate,
    pub source: RateSource,
    pub detail: String,
    pub warning: Option<String>,
}

impl MediaInfo {
    pub fn playback_duration(&self) -> f64 {
        self.container_rate.time_of(self.frame_count as f64)
    }
}

/// Picks the capture rate: `override_rate`, then camera metadata, then the container rate
/// (with a warning, since slow-motion footage would make that wrong).
pub fn resolve_capture_rate(override_rate: Option<FrameRate>, info: &MediaInfo) -> CaptureRate {
    if let Some(rate) = override_rate {
        let warning = info
            .camera_rate
            .as_ref()
            .filter(|c| c.rate != rate)
            .map(|c| {
                format!(
                    "--capture-fps {} overrides {} from {} metadata",
                    rate, c.rate, c.source
                )
            });
        return CaptureRate {
            rate,
            source: RateSource::Override,
            detail: "--capture-fps".into(),
            warning,
        };
    }
    if let Some(c) = &info.camera_rate {
        return CaptureRate {
            rate: c.rate,
            source: RateSource::CameraMetadata,
            detail: format!("{} ({})", c.source, c.raw),
            warning: None,
        };
    }
    CaptureRate {
        rate: info.container_rate,
        source: RateSource::Container,
        detail: "container frame rate".into(),
        warning: Some(format!(
            "no capture-rate metadata found; assuming the container rate of {} is real time — \
             pass --capture-fps if this footage is slowed down",
            info.container_rate
        )),
    }
}

/// Runs ffprobe and the camera-metadata probes on `path`.
pub fn probe(path: &Path) -> Result<MediaInfo> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg(
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,\
             bits_per_raw_sample,color_transfer,duration:format=duration,size",
        )
        .args(["-of", "json"])
        .arg(path)
        .output()
        .context("failed to run ffprobe — is FFmpeg installed and on PATH?")?;
    if !out.status.success() {
        bail!(
            "ffprobe failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let json: Value = serde_json::from_slice(&out.stdout).context("parsing ffprobe output")?;
    let stream = json["streams"]
        .get(0)
        .with_context(|| format!("{} has no video stream", path.display()))?;
    let format = &json["format"];

    let text = |v: &Value| {
        v.as_str()
            .map(str::to_string)
            .or_else(|| v.as_u64().map(|n| n.to_string()))
    };
    let num = |v: &Value| text(v).and_then(|s| s.parse::<u64>().ok());

    let container_rate = [&stream["r_frame_rate"], &stream["avg_frame_rate"]]
        .into_iter()
        .filter_map(|v| v.as_str()?.parse::<FrameRate>().ok())
        .next()
        .context("ffprobe reported no frame rate")?;
    let pix_fmt = text(&stream["pix_fmt"]).unwrap_or_default();
    let bit_depth = num(&stream["bits_per_raw_sample"])
        .map(|b| b as u32)
        .unwrap_or_else(|| bit_depth_of(&pix_fmt));
    let frame_count = match num(&stream["nb_frames"]) {
        Some(n) => n as usize,
        None => {
            let duration: f64 = text(&stream["duration"])
                .or_else(|| text(&format["duration"]))
                .and_then(|s| s.parse().ok())
                .context("ffprobe reported neither frame count nor duration")?;
            (duration * container_rate.fps()).round() as usize
        }
    };

    let (camera, camera_rate) = probe_camera(path)?;
    Ok(MediaInfo {
        path: path.display().to_string(),
        file_size: num(&format["size"])
            .unwrap_or_else(|| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)),
        codec: text(&stream["codec_name"]).unwrap_or_default(),
        width: num(&stream["width"]).context("missing width")? as usize,
        height: num(&stream["height"]).context("missing height")? as usize,
        pix_fmt,
        bit_depth,
        color_transfer: text(&stream["color_transfer"]),
        frame_count,
        container_rate,
        camera,
        camera_rate,
    })
}

/// Bits per component from an FFmpeg pixel format name: >8-bit formats carry an explicit
/// endianness suffix (`yuv420p10le`, `p010le`, `gray16be`); everything else is 8-bit.
fn bit_depth_of(pix_fmt: &str) -> u32 {
    let Some(stem) = pix_fmt
        .strip_suffix("le")
        .or_else(|| pix_fmt.strip_suffix("be"))
    else {
        return 8;
    };
    let digits = stem.len() - stem.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    stem[stem.len() - digits..]
        .parse()
        .ok()
        .filter(|&b| b > 8 && b <= 16)
        .unwrap_or(8)
}

/// Camera-specific capture-rate metadata. Implementations should return `Ok(None)` when the
/// file is not from their camera family.
pub trait CaptureRateProbe {
    fn name(&self) -> &'static str;
    fn probe(&self, path: &Path) -> Result<Option<(Option<String>, CameraRate)>>;
}

/// All known probes, tried in order.
pub fn camera_probes() -> Vec<Box<dyn CaptureRateProbe>> {
    vec![Box::new(SonyNrtProbe)]
}

fn probe_camera(path: &Path) -> Result<(Option<String>, Option<CameraRate>)> {
    for p in camera_probes() {
        if let Some((camera, rate)) = p
            .probe(path)
            .with_context(|| format!("{} probe", p.name()))?
        {
            return Ok((camera, Some(rate)));
        }
    }
    Ok((None, None))
}

/// Sony XAVC S files carry a `NonRealTimeMeta` XML document (in a top-level `meta` box)
/// whose `VideoFrame@captureFps` is the true sensor rate in S&Q / HFR modes.
pub struct SonyNrtProbe;

impl CaptureRateProbe for SonyNrtProbe {
    fn name(&self) -> &'static str {
        "Sony NonRealTimeMeta"
    }

    fn probe(&self, path: &Path) -> Result<Option<(Option<String>, CameraRate)>> {
        let Some(xml) = find_mp4_text(path, "<NonRealTimeMeta", "</NonRealTimeMeta>")? else {
            return Ok(None);
        };
        let Some(raw) = xml_attr(&xml, "VideoFrame", "captureFps") else {
            return Ok(None);
        };
        let rate = raw
            .parse::<FrameRate>()
            .with_context(|| format!("captureFps {raw:?}"))?;
        let camera = match (
            xml_attr(&xml, "Device", "manufacturer"),
            xml_attr(&xml, "Device", "modelName"),
        ) {
            (Some(m), Some(n)) => Some(format!("{m} {n}")),
            (m, n) => m.or(n),
        };
        Ok(Some((
            camera,
            CameraRate {
                rate,
                source: self.name().into(),
                raw,
            },
        )))
    }
}

/// Searches the top-level boxes of an ISO-BMFF (MP4/MOV) file, other than the media data,
/// for a text span starting with `start` and ending with `end`.
fn find_mp4_text(path: &Path, start: &str, end: &str) -> Result<Option<String>> {
    const MAX_BOX: u64 = 16 << 20;
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let len = f.metadata()?.len();
    let mut off = 0u64;
    while off + 8 <= len {
        f.seek(SeekFrom::Start(off))?;
        let mut hdr = [0u8; 16];
        let n = f.read(&mut hdr)?;
        if n < 8 {
            break;
        }
        let size32 = u32::from_be_bytes(hdr[0..4].try_into().unwrap()) as u64;
        let kind = &hdr[4..8];
        let size = match size32 {
            0 => len - off,
            1 if n >= 16 => u64::from_be_bytes(hdr[8..16].try_into().unwrap()),
            1 => break,
            s => s,
        };
        if size < 8 {
            break; // corrupt; stop rather than loop forever
        }
        if kind != b"mdat" && kind != b"free" && kind != b"skip" && size <= MAX_BOX {
            let mut buf = vec![0u8; size as usize];
            f.seek(SeekFrom::Start(off))?;
            f.read_exact(&mut buf)?;
            if let Some(text) = find_span(&buf, start.as_bytes(), end.as_bytes()) {
                return Ok(Some(String::from_utf8_lossy(text).into_owned()));
            }
        }
        off += size;
    }
    Ok(None)
}

fn find_span<'a>(buf: &'a [u8], start: &[u8], end: &[u8]) -> Option<&'a [u8]> {
    let s = buf.windows(start.len()).position(|w| w == start)?;
    let e = buf[s..].windows(end.len()).position(|w| w == end)?;
    Some(&buf[s..s + e + end.len()])
}

/// Value of attribute `attr` on the first `<tag ...>` element. Enough for the flat,
/// machine-written XML cameras emit; not a general XML parser.
fn xml_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let open = format!("<{tag}");
    let mut rest = xml;
    let elem = loop {
        let i = rest.find(&open)?;
        let after = &rest[i + open.len()..];
        if after.starts_with(|c: char| c.is_whitespace() || c == '>' || c == '/') {
            break &after[..after.find('>')?];
        }
        rest = after;
    };
    let key = format!("{attr}=\"");
    let mut search = elem;
    loop {
        let i = search.find(&key)?;
        if i == 0 || search[..i].ends_with(char::is_whitespace) {
            let v = &search[i + key.len()..];
            return Some(v[..v.find('"')?].to_string());
        }
        search = &search[i + key.len()..];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn sony_probe_reads_capture_rate_from_header_fixture() {
        let (camera, rate) = SonyNrtProbe
            .probe(&fixture("sony_rx100m5a_1000p_header.mp4"))
            .unwrap()
            .unwrap();
        assert_eq!(rate.rate, FrameRate::new(1000, 1).unwrap());
        assert_eq!(rate.raw, "1000p");
        assert_eq!(camera.as_deref(), Some("Sony DSC-RX100M5A"));
    }

    #[test]
    fn xml_attr_handles_other_rates_and_lookalike_tags() {
        let xml = r#"<NonRealTimeMeta><VideoFrameX captureFps="1p"/>
            <VideoFrame videoCodec="AVC" xcaptureFps="2p" captureFps="240p" formatFps="30p"/></NonRealTimeMeta>"#;
        assert_eq!(
            xml_attr(xml, "VideoFrame", "captureFps").as_deref(),
            Some("240p")
        );
        assert_eq!(
            xml_attr(xml, "VideoFrame", "formatFps").as_deref(),
            Some("30p")
        );
        assert_eq!(xml_attr(xml, "Device", "modelName"), None);
        for raw in ["240p", "1000p", "119.88p", "500p", "250p"] {
            assert!(raw.parse::<FrameRate>().is_ok(), "{raw}");
        }
    }

    #[test]
    fn non_sony_file_yields_none() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.mov");
        // ftyp box + an empty 64-bit-size mdat.
        let mut bytes = vec![
            0, 0, 0, 16, b'f', b't', b'y', b'p', b'q', b't', b' ', b' ', 0, 0, 0, 0,
        ];
        bytes.extend_from_slice(&[0, 0, 0, 1, b'm', b'd', b'a', b't', 0, 0, 0, 0, 0, 0, 0, 16]);
        std::fs::write(&p, bytes).unwrap();
        assert!(SonyNrtProbe.probe(&p).unwrap().is_none());
    }

    fn info(camera_rate: Option<FrameRate>) -> MediaInfo {
        MediaInfo {
            path: "x".into(),
            file_size: 0,
            codec: "h264".into(),
            width: 1920,
            height: 1080,
            pix_fmt: "yuv420p".into(),
            bit_depth: 8,
            color_transfer: None,
            frame_count: 3720,
            container_rate: FrameRate::new(50, 1).unwrap(),
            camera: None,
            camera_rate: camera_rate.map(|rate| CameraRate {
                rate,
                source: "Sony NonRealTimeMeta".into(),
                raw: "1000p".into(),
            }),
        }
    }

    #[test]
    fn resolution_order_and_warnings() {
        let k = FrameRate::new(1000, 1).unwrap();
        let r = resolve_capture_rate(None, &info(Some(k)));
        assert_eq!(
            (r.rate, r.source.clone(), r.warning.is_none()),
            (k, RateSource::CameraMetadata, true)
        );

        let r = resolve_capture_rate(None, &info(None));
        assert_eq!(
            (r.rate.fps(), r.source.clone()),
            (50.0, RateSource::Container)
        );
        assert!(r.warning.unwrap().contains("--capture-fps"));

        let o = FrameRate::new(240, 1).unwrap();
        let r = resolve_capture_rate(Some(o), &info(Some(k)));
        assert_eq!((r.rate, r.source.clone()), (o, RateSource::Override));
        assert!(
            r.warning.is_some(),
            "override disagreeing with metadata should warn"
        );
        assert!(
            resolve_capture_rate(Some(k), &info(Some(k)))
                .warning
                .is_none()
        );
    }

    #[test]
    fn bit_depth_from_pix_fmt() {
        assert_eq!(bit_depth_of("yuv420p"), 8);
        assert_eq!(bit_depth_of("yuv420p10le"), 10);
        assert_eq!(bit_depth_of("p010le"), 10);
        assert_eq!(bit_depth_of("gray16le"), 16);
        assert_eq!(bit_depth_of("nv12"), 8);
        assert_eq!(bit_depth_of("gray"), 8);
    }
}
