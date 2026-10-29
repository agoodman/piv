# PIVKit (Swift + Metal)

GPU planar image velocimetry for macOS: the same analysis as the Rust `piv` tool (FFT
cross-correlation, multi-pass with sub-pixel window offset, normalized median validation,
temporal smoothing, vorticity and kinetic energy), with every per-pair stage on the GPU.
This directory holds:

- **`Sources/PIVKit`**: the library. Its manifest is [`Package.swift`](../Package.swift) at
  the repository root, so the package can be added by git URL.
- **`Sources/PIVKitCLI`**: the `pivkit` command-line tool.
- **`App/`**: the PIV Mac app.
- **`Tests/`**: unit tests, GPU tests, and parity tests against the Rust reference.

See [`docs/phase3-plan.md`](../docs/phase3-plan.md) for the design, decisions and measured
results.

## Requirements

- macOS 15 or later, Apple silicon (Metal; MPSGraph for the FFT at 64 px and above)
- Xcode 16 or later (Swift 6). Developed with Xcode 26.6 on an M5 Max.
- No FFmpeg or HDF5: decoding uses AVFoundation and output is `.npy` + `run.json`.

## Build and test

From the repository root:

```sh
swift build -c release                    # .build/release/pivkit
swift test                                # unit, GPU and Rust-parity tests
PIV_CLIPS=$PWD/clips swift test           # also tests against clips/C0141.MP4

xcodebuild -project swift/App/PIV.xcodeproj -scheme PIV build   # the app
xcodebuild -project swift/App/PIV.xcodeproj -scheme PIV test    # app model tests
```

Or open `swift/App/PIV.xcodeproj` in Xcode and run the PIV scheme.

## The app

Open a clip (⌘O, or drop it on the window).

- **Setup:**
  - A frame scrubber sets the start and end of the range; dragging on the frame draws the
    region of interest.
  - The capture rate is prefilled from camera metadata (Sony `NonRealTimeMeta`), with a
    warning when only the container rate is known.
  - Scale is either a measured px/mm or a field of view (degrees, axis) plus the
    camera-to-sheet distance.
  - Temporal smoothing is a toggle. Everything else is under Advanced.
  - Settings are remembered per clip.
- **Running:** progress, pairs/s, time remaining and a live quiver preview. Cancel at any
  time.
- **Result:** the quiver movie (smoothed fields when smoothing is on), with the pair's mean
  vorticity, |ω|, kinetic energy and coverage overlaid as it plays. Charts of mean vorticity
  and total KE show a cursor; click to seek. Export Movie… saves the `.mov`.

## `pivkit`

```sh
pivkit info clips/C0141.MP4
pivkit analyze clips/C0141.MP4 -o out/run1 --start-s 0.2 --end-s 1.2 --stride 5 \
    --fov-deg 84 --sheet-distance 0.3 --temporal 9 --video
pivkit bench                      # kernel and per-pair timings on a synthetic 1080p pair
pivkit frame clips/C0141.MP4 250 -o frame250.raw   # one decoded Y plane, for checking
```

`analyze` takes the same options as `piv analyze` (frames, stride, ROI, background,
preprocessing, windows, mask, validation, scale), plus `--temporal N` for smoothing,
`--video` for `<output>_quiver.mov`, and `--path` to choose the correlation kernels. Output
is a directory in the layout of the Rust tool's HDF5 file: `dx.npy`, `dy.npy`, `u.npy`,
`v.npy` (`u_mps.npy`, `v_mps.npy` with a scale), `peak_ratio.npy`, `flags.npy`,
`grid_x/y.npy`, `frame_a/b.npy`, `t.npy`, `stats_*.npy` and `run.json`. With smoothing, the
smoothed fields and `support.npy` are at the top level and the raw fields under `raw/`.

## The library

```swift
import PIVKit

let session = try PIVSession()
var request = AnalysisRequest(source: url)
request.stride = 5
request.scale = ScaleSpec(fovDegrees: 84, fovAxis: .diagonal, sheetDistanceMetres: 0.3)
request.temporalSmoothing = TemporalSmoothing(window: 9)

for try await event in session.analyze(request, render: RenderOptions()) {
    switch event {
    case .progress(let p): print(p.stage, p.done, p.total)
    case .preview(let frame): show(frame.image)             // CGImage, ~2 per second
    case .analysed(let result): use(result.fields, result.stats)
    case .finished(_, let movie): play(movie!)
    }
}
```

Cancelling the consuming task stops the run. Lower-level pieces (`MediaInfo`,
`VideoDecoder`, `FramePreparer`, `PIVAnalyzer`, `QuiverRenderer`, `analyzeRun`,
`renderQuiverMovie`) are public for tools and tests.

### Pipeline

| Stage | Where |
|---|---|
| Decode the stored Y plane (or green) | AVFoundation / VideoToolbox, own thread |
| Upload, background subtraction, Gaussian blur, local normalization | Metal |
| Window placement from the previous pass (predictor, symmetric split) | Metal |
| Window loading, signal mask, cross-correlation | Metal: fused kernel ≤ 32 px, custom mixed-radix FFT at 48 px, MPSGraph FFT ≥ 64 px; two windows share each inverse FFT |
| Peak, peak ratio, Gaussian sub-pixel fit | Metal |
| Peak-ratio / range / normalized median tests, replacement | Metal |
| Flow statistics, temporal smoothing | CPU |
| Quiver frames → movie | Metal → AVAssetWriter |

All passes of a pair are one command buffer; the CPU reads back only the final field.

## Agreement with the Rust tool

Parity is tested stage by stage against reference cases written by
`cargo run -p piv-core --example fixtures` (in `rust/`; it regenerates
`Tests/PIVKitTests/Fixtures/cases`).

- **Window placement and validation:** bit for bit (FMA contraction is off in the kernels).
- **Correlation peaks:** within 3e-6 px.
- **Preprocessing:** within 3.5e-5.
- **Flags:** identical.

On a real clip (C0141, 995 pairs), 10 of 1.88 million flags differ and the median vector
difference is 5e-7 px. Compare runs with `rust/scripts/compare_runs.py`.

Three deliberate differences from the Rust defaults:

- **Luma.** PIVKit reads the stored Y plane. `piv --channel luma` uses FFmpeg's gray
  conversion, which on the sample clips mixes in chroma and is ~1.6× darker; compare against
  `piv --channel yplane`.
- **Mask thresholds.** Because Y is brighter, the default mask is Rust's ×1.6
  (`min_mean 0.064`, `min_std 0.024`), which reproduces `piv analyze`'s no-signal fraction on
  C0141 (63 %).
- **Output.** No HDF5 yet (`.npy` from `pivkit`; the app saves the movie).

## Performance (M5 Max, 1920×1080, defaults 96 → 64 → 64 px)

| | Rate |
|---|---|
| Full C0141 (3 715 pairs), temporal 9 | 177 pairs/s (20.9 s) |
| Quiver movie render + encode | ~430 frames/s |
| Rust `piv analyze` on the same machine, same day | 17–19 pairs/s |
