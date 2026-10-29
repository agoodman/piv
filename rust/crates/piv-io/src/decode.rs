//! Frame decoding via an `ffmpeg` subprocess streaming raw grayscale frames.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use piv_core::{Frame, FrameRate};
use serde::{Deserialize, Serialize};

/// Which image plane to analyse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    /// FFmpeg's `gray` conversion of the decoded frame. Despite the name this is not the Y
    /// plane as stored: on the Sony clips (BT.709, limited range) it mixes in chroma and comes
    /// out ~37 % darker on average than Y (C0141 frame 0: mean 23.3 vs 37.2 DN).
    #[default]
    Luma,
    /// The green channel after conversion to RGB (best match for 532 nm light).
    Green,
    /// The Y plane exactly as stored (`extractplanes=y`), with no range or colour conversion.
    /// This is what PIVKit (Swift) reads.
    Yplane,
}

/// Region of interest in full-frame pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Roi {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl std::str::FromStr for Roi {
    type Err = String;

    /// Parses `x,y,w,h`.
    fn from_str(s: &str) -> Result<Self, String> {
        let v: Vec<usize> = s
            .split(',')
            .map(|p| {
                p.trim()
                    .parse()
                    .map_err(|_| format!("invalid ROI {s:?}; expected x,y,w,h"))
            })
            .collect::<Result<_, _>>()?;
        match v[..] {
            [x, y, width, height] if width > 0 && height > 0 => Ok(Roi {
                x,
                y,
                width,
                height,
            }),
            _ => Err(format!("invalid ROI {s:?}; expected x,y,w,h with w,h > 0")),
        }
    }
}

/// A source of sequentially numbered grayscale frames.
pub trait FrameSource {
    /// Next frame and its index in the source, or `None` at the end.
    fn next_frame(&mut self) -> Result<Option<(usize, Frame)>>;
    /// Height and width of the frames produced.
    fn dims(&self) -> (usize, usize);
}

#[derive(Debug, Clone)]
pub struct DecodeOptions {
    /// Index of the first frame to decode.
    pub start: usize,
    /// Number of frames to decode (`None` = to the end).
    pub count: Option<usize>,
    pub channel: Channel,
    pub roi: Option<Roi>,
    /// Decode at 16 bits per sample (for sources deeper than 8 bits).
    pub high_bit_depth: bool,
    /// Container (stream) frame rate, used to seek to `start` by timestamp.
    pub container_rate: FrameRate,
    /// Full frame size, used to validate the ROI.
    pub frame_size: (usize, usize),
}

/// Streams frames from `ffmpeg -f rawvideo`.
pub struct FfmpegSource {
    child: Child,
    stdout: BufReader<ChildStdout>,
    path: PathBuf,
    height: usize,
    width: usize,
    bytes_per_sample: usize,
    buf: Vec<u8>,
    next_index: usize,
    remaining: Option<usize>,
}

impl FfmpegSource {
    pub fn open(path: &Path, opts: &DecodeOptions) -> Result<Self> {
        let (full_h, full_w) = opts.frame_size;
        let (x, y, width, height) = match opts.roi {
            Some(r) => {
                if r.x + r.width > full_w || r.y + r.height > full_h {
                    bail!("ROI {r:?} exceeds the {full_w}×{full_h} frame");
                }
                (r.x, r.y, r.width, r.height)
            }
            None => (0, 0, full_w, full_h),
        };

        let (pix, bytes_per_sample) = if opts.high_bit_depth {
            ("gray16le", 2)
        } else {
            ("gray", 1)
        };
        let mut filters = Vec::new();
        // Take Y before cropping, so odd ROI offsets don't interact with chroma subsampling.
        if opts.channel == Channel::Yplane {
            filters.push("extractplanes=y".into());
        }
        if opts.roi.is_some() {
            filters.push(format!("crop={width}:{height}:{x}:{y}"));
        }
        if opts.channel == Channel::Green {
            filters.push(format!(
                "format={}",
                if opts.high_bit_depth {
                    "gbrp16le"
                } else {
                    "gbrp"
                }
            ));
            filters.push("extractplanes=g".into());
        }
        filters.push(format!("format={pix}"));

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-nostdin", "-hide_banner", "-v", "error"]);
        if opts.start > 0 {
            // Seek half a frame early: input seeking decodes from the preceding keyframe and
            // drops frames before this timestamp, so the first frame emitted is `start`.
            let t = opts.container_rate.time_of(opts.start as f64 - 0.5);
            cmd.args(["-ss", &format!("{t:.6}")]);
        }
        cmd.arg("-i").arg(path).args([
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &filters.join(","),
        ]);
        if let Some(n) = opts.count {
            cmd.args(["-frames:v", &n.to_string()]);
        }
        cmd.args([
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "-",
        ]);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .context("failed to start ffmpeg — is FFmpeg installed and on PATH?")?;
        let stdout = BufReader::with_capacity(1 << 22, child.stdout.take().expect("piped"));
        Ok(Self {
            child,
            stdout,
            path: path.to_path_buf(),
            height,
            width,
            bytes_per_sample,
            buf: vec![0; width * height * bytes_per_sample],
            next_index: opts.start,
            remaining: opts.count,
        })
    }

    fn fail(&mut self) -> anyhow::Error {
        let _ = self.child.kill();
        let mut err = String::new();
        if let Some(mut e) = self.child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        let _ = self.child.wait();
        anyhow::anyhow!(
            "ffmpeg failed decoding {}: {}",
            self.path.display(),
            err.trim()
        )
    }
}

impl FrameSource for FfmpegSource {
    fn next_frame(&mut self) -> Result<Option<(usize, Frame)>> {
        if self.remaining == Some(0) {
            return Ok(None);
        }
        let mut filled = 0;
        while filled < self.buf.len() {
            match self.stdout.read(&mut self.buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e).context("reading from ffmpeg"),
            }
        }
        if filled == 0 {
            let status = self.child.wait()?;
            if !status.success() {
                return Err(self.fail());
            }
            return Ok(None);
        }
        if filled < self.buf.len() {
            return Err(self.fail().context("truncated frame"));
        }

        let data: Vec<f32> = if self.bytes_per_sample == 1 {
            self.buf.iter().map(|&b| b as f32 / 255.0).collect()
        } else {
            self.buf
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0)
                .collect()
        };
        let frame =
            Frame::from_shape_vec((self.height, self.width), data).expect("buffer sized to frame");
        let index = self.next_index;
        self.next_index += 1;
        if let Some(r) = &mut self.remaining {
            *r -= 1;
        }
        Ok(Some((index, frame)))
    }

    fn dims(&self) -> (usize, usize) {
        (self.height, self.width)
    }
}

impl Drop for FfmpegSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Decodes the single frame at `index`.
pub fn read_frame(path: &Path, index: usize, opts: &DecodeOptions) -> Result<Frame> {
    let opts = DecodeOptions {
        start: index,
        count: Some(1),
        ..opts.clone()
    };
    FfmpegSource::open(path, &opts)?
        .next_frame()?
        .map(|(_, f)| f)
        .with_context(|| format!("frame {index} not found in {}", path.display()))
}
