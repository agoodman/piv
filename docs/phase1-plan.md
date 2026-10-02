# Phase 1 plan: frame-to-frame PIV in Rust

## 1. Context

We want a fast, command-line planar velocimetry library that reads high-speed footage of a 532 nm laser light sheet seeded with dry-ice fog. It should estimate how the image pattern moves between frames and use the true frame timing to turn that motion into a 2D velocity vector field. Phase 1 builds the core pipeline end to end: decode → preprocess → cross-correlate → validate → write vector fields and quiver overlays. Results are in **pixel units only** (px/frame, px/s). Physical calibration comes in a later phase.

## 2. What the sample clip tells us (`clips/C0141.MP4`)

I read these from the MP4 container and Sony's embedded `NonRealTimeMeta` XML, and I pulled sample frames:

| Property | Value | Consequence |
|---|---|---|
| Camera | Sony DSC-RX100M5A | Consumer HFR camera, not a scientific camera |
| Recording mode | `slowAndQuickMotion`, **captureFps = 1000p**, formatFps = 50p | **The container says 50 fps but the real interval is 1 ms.** If we trust the container, velocities come out 20× too low. |
| Frames | 3720 (74.4 s at 50p = **3.72 s of real time**) | |
| Stream | H.264 High@L4.2, 1920×1080, ~50 Mb/s, 8-bit 4:2:0, Rec.709 | Lossy compression; chroma is subsampled |
| Image content | Dense fog with smooth vortical structures; **no individually resolved particles**; noticeably soft (the camera likely upscales its native HFR readout) | This is closer to "smoke image velocimetry" than classic sparse-particle PIV. Correlation has to track texture, so windows must be fairly large (≥32 px). |
| Noise | Mean absolute difference between consecutive frames ≈ 6 DN, roughly constant | Strong sensor noise. Per-frame motion at 1 ms is only a few px, so using pairs further apart (stride > 1) will often improve signal-to-noise. |
| Background | Mostly black outside the sheet; bright glare band in the lower right | Mask low-signal regions and flag their vectors invalid instead of reporting noise as motion |

Consecutive decoded frames were all distinct (no duplicated frames from the HFR buffer), so frame *n* → *n+k* really is *k* ms.

## 3. Ecosystem research: existing Rust tooling

- **No maintained Rust PIV crate exists.** Mature PIV tools are Python or MATLAB (OpenPIV, PIVlab, TorchPIV). We'll implement the algorithms ourselves, using the open literature and OpenPIV/PIVlab as references.
- The building blocks are mature:
  - `rustfft` / `realfft`: FFTs with cached plans and SIMD on NEON/AVX.
  - `ndarray`: 2D arrays. `rayon`: data parallelism across interrogation windows.
  - Video decoding: `ffmpeg-next` is in-process but maintenance-only and needs ~40 lines of decode loop. `video-rs` is a higher-level wrapper around it but still a work in progress. `ffmpeg-sidecar` pipes raw frames from an `ffmpeg` subprocess and is actively maintained.
  - `ndarray-npy` writes `.npy`/`.npz` with no C dependencies. `hdf5-metno` writes HDF5 but needs the system libhdf5.
  - `image` + `imageproc` draw overlays. `clap` provides the CLI.
- Rust beats C++ here because rayon + rustfft give us safe, simple parallel FFTs, cargo makes the build simple, and `pyo3` leaves a clean path to Python bindings later.

## 4. Architecture

A Cargo workspace keeps the algorithm core free of I/O dependencies:

```
Cargo.toml                (workspace)
crates/
  piv-core/               pure algorithms, no I/O deps
    src/lib.rs
    src/frame.rs          Frame (ndarray::Array2<f32>), FramePair, timing (dt)
    src/preprocess.rs     background subtraction, local normalization, masking
    src/grid.rs           interrogation window grid (size, overlap, multi-pass)
    src/correlate.rs      FFT cross-correlation, plan cache, subpixel peak fit
    src/multipass.rs      iterative window-offset refinement
    src/validate.rs       normalized median test, peak-ratio test, replacement
    src/field.rs          VectorField {x, y, u, v, peak_ratio, flags}
  piv-io/
    src/decode.rs         FrameSource trait + FFmpeg-backed implementation
    src/metadata.rs       capture-fps discovery (Sony NRT XML, container, override)
    src/write.rs          .npz writer (+ HDF5 behind `hdf5` cargo feature)
    src/render.rs         quiver overlay PNGs; MP4 assembly via ffmpeg
  piv-cli/
    src/main.rs           `piv` binary (clap)
tests/fixtures/           small synthetic images
docs/phase1-plan.md
```

## 5. Pipeline details

### 5.1 Decoding (`piv-io::decode`)
- A `FrameSource` trait with methods `next_frame() -> Option<(index, Frame)>`, `seek(index)`, and `len()`.
- First implementation: **`ffmpeg-sidecar`**. It streams raw `gray` (the Y/luma plane), so we skip RGB conversion; luma is ~59 % green-weighted, which suits 532 nm light. A `--channel green` option uses `format=gbrp,extractplanes=g` instead.
- A sliding ring buffer of `stride + 1` frames, so memory stays constant for arbitrarily long clips.
- Using a trait means we can later swap in in-process `ffmpeg-next` decoding, or image-sequence input (TIFF/PNG folders from scientific cameras), without touching the core.

### 5.2 Timing (`piv-io::metadata`) — the main correctness risk
Capture fps resolves in this order, and the chosen source is recorded in the output metadata:
1. `--capture-fps <f>` CLI override.
2. Sony `NonRealTimeMeta` XML (`VideoFrame@captureFps`), parsed from the MP4 `meta` box.
3. Container frame rate (`stts`/`mdhd`), with a **loud warning** that HFR footage may be time-stretched.

Then `dt = stride / capture_fps` and `velocity_px_per_s = displacement_px / dt`.

### 5.3 Preprocessing (`piv-core::preprocess`)
- Optional background subtraction: a rolling temporal minimum or mean over N frames removes static glare, such as the bright band in the lower right.
- Local intensity normalization (subtract the local mean and divide by the local std over a sliding window). This matters for fog, where brightness varies a lot across the sheet.
- Signal mask: windows whose std or mean falls below a threshold are marked `NO_SIGNAL` and not correlated.

### 5.4 Cross-correlation (`piv-core::correlate`, `multipass`)
- Standard FFT cross-correlation of zero-mean windows, normalized so peak heights can be compared.
- **Multi-pass with integer window offset:** pass 1 uses large windows (default 64 px), and later passes (default down to 32 px, 50 % overlap) shift the second window by the rounded predictor from the previous pass. This handles displacements larger than ¼ of the window.
- 3-point Gaussian sub-pixel peak fit, falling back to parabolic fit when values are ≤ 0.
- Per-vector quality: primary-to-secondary peak ratio (PPR).
- Parallelism: rayon over windows, one FFT plan cache per thread, and real-to-complex FFTs (`realfft`).
- *Deferred to phase 2:* continuous window deformation (WIDIM), which matters for the strongly sheared vortices visible in this clip.

### 5.5 Validation (`piv-core::validate`)
- Normalized median test (Westerweel & Scarano 2005: 3×3 neighbourhood, ε = 0.1 px, threshold 2).
- PPR threshold (default 1.3).
- Invalid vectors are replaced by the median of valid neighbours and **flagged, not hidden**. The flags are a bitfield: `NO_SIGNAL`, `LOW_PPR`, `MEDIAN_OUTLIER`, `REPLACED`.

### 5.6 Output (`piv-io::write`, `render`)
- **NPZ, one file per frame pair:** `x`, `y`, `u`, `v` (px/s), `dx`, `dy` (px), `peak_ratio`, `flags`. A `run.json` sidecar holds every parameter, the capture fps and its source, dt, stride, frame indices, and the tool version.
- **HDF5 (cargo feature `hdf5`, needs `brew install hdf5`):** a single file holding all pairs as one time series with the same fields and attributes.
- **Quiver overlays:** PNGs of each frame with arrows coloured by magnitude and invalid vectors drawn in a distinct colour. With `--video`, the PNGs are assembled into an MP4 by an ffmpeg subprocess.

## 6. CLI

```
piv info    clips/C0141.MP4
            # prints resolution, frame count, container fps, capture fps + source

piv analyze clips/C0141.MP4 -o out/run1 \
            --start 200 --end 400 --stride 2 \
            --window 64,32 --overlap 0.5 \
            [--capture-fps 1000] [--channel luma|green] \
            [--background min:50] [--format npz|hdf5] \
            [--quiver] [--video] [--roi x,y,w,h]
```

`--start`/`--end`/`--stride` let you process arbitrary sub-ranges and frame separations.

## 7. Verification

1. **Unit tests (`piv-core`)** on synthetic image pairs, generated in-test from a seeded RNG (random Gaussian particles plus a smooth fog-like noise texture):
   - Uniform integer and sub-pixel shifts (0.25, 0.5, 3.7 px): RMS error < 0.05 px.
   - Large shift (20 px with 32 px final windows) is recovered through multi-pass.
   - Solid-body rotation and linear shear: error is bounded and vectors don't point the wrong way.
   - Injected outliers are caught by the median test.
   - Blank windows get `NO_SIGNAL`.
2. **Metadata test:** the parser returns `captureFps = 1000` for a small MP4 header fixture extracted from C0141, and container fallback produces a warning.
3. **End-to-end smoke test** (ignored by default because it needs the clip): `piv analyze clips/C0141.MP4 --start 200 --end 260 --quiver --video`.
   - Check that the run finishes, the vector count matches the grid, and the valid-vector fraction is reasonable in seeded regions.
   - Visually check that the quiver video follows the vortex motion.
   - Sanity check: displacements at stride 1 are a few px/frame, consistent with what the sample frames show.
4. **Performance benchmark** (`criterion`): one 1920×1080 pair at 32 px / 50 % overlap. Target < 100 ms on Apple Silicon; full clip (3719 pairs) in a few minutes.
5. `cargo clippy -- -D warnings` and `cargo fmt --check` stay clean.

## 8. Milestones (commits on `phase-1`)

1. Workspace scaffold, CI-free `cargo test` green, `.gitignore`.
2. `piv-core`: correlation + subpixel + synthetic tests.
3. Multi-pass + validation + tests.
4. `piv-io`: decoding, metadata/timing, `piv info`.
5. Writers (NPZ, HDF5 feature) + `piv analyze`.
6. Quiver rendering + video.
7. Benchmark, tuning on C0141, README usage section.

## 9. Setup and repo hygiene

- Toolchain: `rustup` (stable), plus `brew install ffmpeg` (runtime for decoding and video). HDF5 is only needed for `--features hdf5`. None of these are installed on this machine yet.
- `.gitignore`: `.DS_Store`, `target/`, `out/`, `clips/*.MP4`. The 482 MB clip stays local and untracked; Git LFS is an option if you want clips versioned.

## 10. Out of scope for phase 1 (candidate phase 2+)
- Physical calibration (px → m via target or known field of view) and lens/perspective correction.
- Window deformation (WIDIM), ensemble/time-averaged correlation, dense optical flow (Horn–Schunck / Farnebäck) as an alternative estimator for fog texture.
- Derived quantities (vorticity, strain, streamlines), GPU acceleration, Python bindings (pyo3).

## 11. Open questions for your review
1. Is 1 ms (1000 fps) right for every clip you'll record, or will you also use 250/500 fps modes? The metadata parser handles all of them, but please confirm the override behaviour you want.
2. Do you know the light-sheet thickness and the camera's field of view? This isn't needed now, but it shapes the phase 2 calibration design.
3. Should the default output be NPZ or HDF5?
