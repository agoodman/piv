# Phase 1 plan: frame-to-frame PIV in Rust

## 1. Context

We want a fast, command-line planar velocimetry library that reads high-speed footage of a 532 nm laser light sheet seeded with dry-ice fog. It should estimate how the image pattern moves between frames and use the true frame timing to turn that motion into a 2D velocity vector field. Phase 1 builds the core pipeline end to end: decode → preprocess → cross-correlate → validate → write vector fields and quiver overlays. Results are always in pixel units (px/frame, px/s). If you supply a field of view and the camera-to-sheet distance, the tool also reports an approximate physical velocity (m/s); see §5.7. Proper target-based calibration comes in a later phase.

Requirements confirmed after review:
- **Any capture frame rate.** Cameras in use run at 1000 fps and 240 fps, but nothing assumes a particular rate.
- **User-entered field of view.** The sample was shot at 84°. A light sheet of about 1 mm is recorded as run metadata.
- **HDF5 is the default output format.**

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
    src/scale.rs          optional px → m scale from FOV + distance (§5.7)
    src/write.rs          HDF5 writer (default) + .npz writer
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
The frame rate is never hard-coded. It is stored as an exact rational number (e.g. 1000/1, 240/1, 240000/1001), so drop-frame rates like 239.76 fps also work. Capture fps resolves in this order, and the chosen source is recorded in the output metadata:
1. `--capture-fps <f>` CLI override. It accepts any positive value (`240`, `1000`, `239.76`, `240000/1001`).
2. Camera metadata. Phase 1 reads Sony `NonRealTimeMeta` XML (`VideoFrame@captureFps`, such as `"1000p"` or `"240p"`) from the MP4 `meta` box. Each camera vendor gets its own small parser behind a `CaptureRateProbe` trait, so other cameras can be added once we have sample files from them.
3. Container frame rate (`stts`/`mdhd`). This is correct for cameras that write high-frame-rate video at its true rate, such as a 240 fps file played back at 240 fps. But it is wrong for slow-motion recordings like C0141, so the tool prints a **warning**: "no capture-rate metadata found; assuming the container rate of N fps is real time — pass --capture-fps if this footage is slowed down."

If the camera metadata and the container rate disagree, `piv info` shows both values.

Then `dt = stride / capture_fps` and `velocity_px_per_s = displacement_px / dt`.

**Choosing the stride for different frame rates.** Good correlation needs a displacement of roughly 2–8 px in the image plane. Particles must also stay inside the light sheet between the two frames, which means out-of-plane motion must be less than about ¼ of the sheet thickness. With a ~1 mm sheet, that limits out-of-plane speed to about 0.25 m/s at stride 1 and 1000 fps, or about 0.06 m/s at 240 fps. `piv analyze` prints the in-plane displacement statistics for each run, so you can tune `--stride`: increase it when motion is sub-pixel, and decrease it when there are many outliers or few valid vectors.

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
- **HDF5 (default), one file per run, written with `hdf5-metno` against the system libhdf5:**
  - `/grid/x` and `/grid/y` (px): window centres, shared by every frame pair.
  - `/fields/dx` and `/fields/dy` (px), `/fields/u` and `/fields/v` (px/s), `/fields/peak_ratio`, and `/fields/flags`. Each has shape `[n_pairs, ny, nx]` and is chunked one pair per chunk, so the file can be written while streaming and read one pair at a time.
  - `/fields/u_mps` and `/fields/v_mps`, only when a physical scale is given (§5.7).
  - `/pairs/frame_a`, `/pairs/frame_b`, and `/pairs/t` (s): the time of each pair's midpoint, measured from the clip start at the capture rate.
  - Root attributes: every run parameter, the capture fps and its source, dt, stride, the scale inputs and result, the light-sheet thickness, the source file path and size, and the tool version. Pixel arrays are `f32`; flags are `u8`.
  - The file can be read directly with `h5py` or MATLAB.
- **NPZ (`--format npz`):** one `.npz` per frame pair plus a `run.json` sidecar, for environments without HDF5.
- **Quiver overlays:** PNGs of each frame with arrows coloured by magnitude and invalid vectors drawn in a distinct colour. With `--video`, the PNGs are assembled into an MP4 by an ffmpeg subprocess.
- **Quiver overlays:** PNGs of each frame with arrows coloured by magnitude and invalid vectors drawn in a distinct colour. With `--video`, the PNGs are assembled into an MP4 by an ffmpeg subprocess.

### 5.7 Optional physical scale (`piv-io::scale`)
Phase 1 does not do true calibration. It can, however, turn a user-entered field of view into an approximate metres-per-pixel scale. This assumes a pinhole camera looking straight at the sheet with no lens distortion:

`m_per_px = 2 · D · tan(FOV / 2) / N_px`

- `D` is the camera-to-sheet distance, `FOV` is the field of view, and `N_px` is the image width, height, or diagonal in pixels, depending on `--fov-axis`.
- Inputs: `--fov-deg 84 --fov-axis diagonal|horizontal|vertical --sheet-distance <m>`.
- The field of view alone is not enough; the distance is also required. If only `--fov-deg` is given, the tool stores it as metadata and outputs pixel units only.
- The default axis is `diagonal`, because lens specs usually quote a diagonal angle; 84° matches the RX100's 24 mm-equivalent wide end.
- **Caveat for this camera:** the RX100's high-frame-rate modes read out a cropped region of the sensor, so the *effective* field of view at 1000 fps is likely narrower than the 84° lens spec. The tool prints this caveat with the scale it computes.
- `--px-per-mm <s>` sets the scale directly. Use it when you measure the scale from an object of known size in the frame; it is the more trustworthy option.
- `--sheet-thickness-mm` (e.g. 1.0) is stored as metadata only and is used in the stride guidance printed per run (§5.2).

## 6. CLI

```
piv info    clips/C0141.MP4
            # prints resolution, frame count, container fps, capture fps + source,
            # and real-time duration

piv analyze clips/C0141.MP4 -o out/run1.h5 \
            --start 200 --end 400 --stride 2 \
            --window 64,32 --overlap 0.5 \
            [--capture-fps 240] [--channel luma|green] \
            [--background min:50] [--format hdf5|npz] \
            [--fov-deg 84 --fov-axis diagonal --sheet-distance 0.5 | --px-per-mm 3.2] \
            [--sheet-thickness-mm 1.0] \
            [--quiver] [--video] [--roi x,y,w,h]
```

- `--start`, `--end` and `--stride` let you process any sub-range of frames and any frame separation.
- `--start` and `--end` take frame indices; `--start-s` and `--end-s` take real (capture-time) seconds instead.

## 7. Verification

1. **Unit tests (`piv-core`)** on synthetic image pairs, generated in-test from a seeded RNG (random Gaussian particles plus a smooth fog-like noise texture):
   - Uniform integer and sub-pixel shifts (0.25, 0.5, 3.7 px): RMS error < 0.05 px.
   - Large shift (20 px with 32 px final windows) is recovered through multi-pass.
   - Solid-body rotation and linear shear: error is bounded and vectors don't point the wrong way.
   - Injected outliers are caught by the median test.
   - Blank windows get `NO_SIGNAL`.
2. **Timing and metadata tests:**
   - The parser returns 1000 fps for a small MP4 header fixture extracted from C0141.
   - The Sony XML parser handles `240p`, `1000p` and other rates.
   - The container fallback produces the warning.
   - Rational parsing of `--capture-fps` (`240`, `239.76`, `240000/1001`) works.
   - `dt` is correct for each stride and frame rate.
3. **Scale tests:** the formula in §5.7 gives the right result for each `--fov-axis` against hand-computed values; `--px-per-mm` overrides the FOV inputs; and FOV without a distance produces pixel units only.
4. **HDF5 round-trip test:** write a small run, read it back with `hdf5-metno`, and check that the datasets, shapes and attributes match. A short Python snippet in the README shows reading the same file with `h5py`.
5. **End-to-end smoke test** (ignored by default because it needs the clip): `piv analyze clips/C0141.MP4 -o out/smoke.h5 --start 200 --end 260 --quiver --video`.
   - Check that the run finishes, the vector count matches the grid, and the valid-vector fraction is reasonable in seeded regions.
   - Visually check that the quiver video follows the vortex motion.
   - Sanity check: displacements at stride 1 are a few px/frame, consistent with what the sample frames show.
6. **Performance benchmark** (`criterion`): one 1920×1080 pair at 32 px / 50 % overlap. Target < 100 ms on Apple Silicon; full clip (3719 pairs) in a few minutes.
7. `cargo clippy -- -D warnings` and `cargo fmt --check` stay clean.

## 8. Milestones (commits on `phase-1`)

1. Workspace scaffold, CI-free `cargo test` green, `.gitignore`.
2. `piv-core`: correlation + subpixel + synthetic tests.
3. Multi-pass + validation + tests.
4. `piv-io`: decoding, metadata/timing (any frame rate), `piv info`.
5. HDF5 writer (default) + NPZ writer + optional scale + `piv analyze`.
6. Quiver rendering + video.
7. Benchmark, tuning on C0141, README usage section.

## 9. Setup and repo hygiene

- Toolchain: `rustup` (stable), plus `brew install ffmpeg hdf5 pkg-config`.
  - ffmpeg is needed at runtime for decoding and video.
  - libhdf5 is needed at build time for `hdf5-metno`, which is a default feature. Building with `--no-default-features` gives an NPZ-only build without HDF5.
  - None of these are installed on this machine yet.
- `.gitignore`: `.DS_Store`, `target/`, `out/`, `clips/*.MP4`. The 482 MB clip stays local and untracked; Git LFS is an option if you want clips versioned.

## 10. Out of scope for phase 1 (candidate phase 2+)
- Target-based physical calibration (dot grid or checkerboard in the sheet plane) and lens/perspective correction. This replaces the approximate FOV scale from §5.7.
- Window deformation (WIDIM), ensemble/time-averaged correlation, dense optical flow (Horn–Schunck / Farnebäck) as an alternative estimator for fog texture.
- Derived quantities (vorticity, strain, streamlines), GPU acceleration, Python bindings (pyo3).

## 11. Decisions from review
1. **Frame rate:** support any capture rate (cameras on hand: 1000 fps and 240 fps), resolved from override → camera metadata → container (§5.2).
2. **Field of view / sheet:** FOV is user-entered (84° for C0141), and an approximate m/s scale is produced only when a distance or `--px-per-mm` is also given (§5.7). The sheet thickness (~1 mm) is recorded and used for stride guidance.
3. **Output:** HDF5 by default, with NPZ optional (§5.6).

## 12. Remaining open questions
1. Is the 84° FOV the lens's diagonal spec, or was it measured for the HFR-cropped frame? And roughly how far was the camera from the sheet?
2. What is the second (240 fps) camera? A sample clip from it would let us add its metadata parser and a test fixture.
