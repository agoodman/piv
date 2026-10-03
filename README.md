# PIV

Planar image velocimetry for high-speed light-sheet video, written in Rust. `piv` reads a clip,
cross-correlates interrogation windows between frame pairs, validates the vectors, and writes
displacement and velocity fields to HDF5 (or NPZ), optionally with quiver overlay images and
video.

See [`docs/phase1-plan.md`](docs/phase1-plan.md) for the design and the findings from the sample
clip.

## Requirements

- Rust (stable, via `rustup`)
- FFmpeg (`ffmpeg` and `ffprobe` on `PATH`) for decoding and video output
- HDF5 and pkg-config for the default HDF5 output: `brew install ffmpeg hdf5 pkg-config`
  - Build without HDF5 (NPZ output only): `cargo build --release --no-default-features`

## Build and test

```sh
cargo build --release            # binary at target/release/piv
cargo test                       # unit + synthetic-image tests
cargo test -- --include-ignored  # also runs tests against clips/C0141.MP4
cargo bench -p piv-core          # 1920×1080 pair benchmark
```

## Usage

```sh
# What's in the file, and what frame rate was it really captured at?
piv info clips/C0141.MP4

# Analyse 1 s of capture time, pairing each frame with the one 5 frames later.
piv analyze clips/C0141.MP4 -o out/run1.h5 --start-s 0.2 --end-s 1.2 --stride 5 \
    --quiver --video \
    --fov-deg 84 --sheet-distance 0.3 --sheet-thickness-mm 1
```

`piv analyze` prints a summary and hints for choosing `--stride`. Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--start/--end`, `--start-s/--end-s` | whole clip | frame range (end exclusive), by index or capture-time seconds |
| `--stride N` | 1 | pair frame *i* with *i + N*; dt = N / capture fps |
| `--pair-step N` | 1 | analyse a pair starting every N frames |
| `--capture-fps R` | from metadata | override capture rate (`240`, `1000`, `239.76`, `240000/1001`) |
| `--window` | `96,64` | window size per pass, largest first |
| `--overlap` | `0.5` | window overlap |
| `--roi x,y,w,h` | full frame | region of interest |
| `--background` | `min:40` | static background from N frames sampled across the clip; `none` to disable |
| `--channel` | `luma` | `luma` or `green` |
| `--format` | `hdf5` | `hdf5` (one file) or `npz` (directory, one file per pair) |
| `--px-per-mm` / `--fov-deg` + `--sheet-distance` | none | optional physical scale for m/s output |

### Frame timing

High-speed cameras often record at one rate and play back at another. `C0141.MP4` is stored as
50 fps but was captured at 1000 fps. `piv` resolves the capture rate from `--capture-fps`, then
camera metadata (Sony `NonRealTimeMeta`), then the container rate, and warns when it falls back
to the container.

### Output (HDF5)

| Dataset | Shape | Units |
|---|---|---|
| `grid/x`, `grid/y` | `[nx]`, `[ny]` | window centres, full-frame px |
| `fields/dx`, `fields/dy` | `[pairs, ny, nx]` | px |
| `fields/u`, `fields/v` | `[pairs, ny, nx]` | px/s |
| `fields/u_mps`, `fields/v_mps` | `[pairs, ny, nx]` | m/s (only with a scale) |
| `fields/peak_ratio`, `fields/flags` | `[pairs, ny, nx]` | flags: 1 no signal, 2 low peak ratio, 4 median outlier, 8 replaced, 16 out of range |
| `pairs/frame_a`, `pairs/frame_b`, `pairs/t` | `[pairs]` | frame indices; pair midpoint time (s) |

Root attributes hold every run parameter (also as `run_json`). Vectors with no signal are NaN.
Replaced vectors keep their original rejection flag plus `REPLACED`.

```python
import h5py, numpy as np
with h5py.File("out/run1.h5") as f:
    x, y = f["grid/x"][:], f["grid/y"][:]
    u, v, flags = f["fields/u"][0], f["fields/v"][0], f["fields/flags"][0]
    good = (flags & (1 | 2 | 4 | 16)) == 0
    print(f.attrs["capture_fps"], f.attrs["dt_s"], np.nanmedian(np.hypot(u, v)[good]))
```

### Smoothing

`piv smooth` post-processes an HDF5 run without re-running the correlation:

```sh
# Compare settings: cross-validation error, coverage, peak flattening, jitter.
piv smooth out/run1.h5 --sweep

# 9-pair temporal median (≈9 ms at 1000 fps), plus a side-by-side raw | smoothed video.
piv smooth out/run1.h5 -o out/run1_t9.h5 --temporal 9 --video
```

| Option | Meaning |
|---|---|
| `--temporal N` | moving filter over N consecutive pairs (odd) at each grid point |
| `--temporal-kernel median\|mean` | median (default) resists leftover outliers; mean removes slightly more noise |
| `--spatial σ` | Gaussian low-pass, σ in grid cells, skipping missing vectors |
| `--min-support N` | raw vectors needed in the temporal window (default N/3) |

Only accepted raw vectors contribute, and outputs are produced only where the window had
seeding in that pair. The output has the same layout as `piv analyze` (`fields/u`, …) with
smoothed values, plus `fields/support` (contributing vectors) and the untouched raw fields under
`raw/`; the settings are in the `smoothing_json` attribute.

The sweep's **CV error** predicts each accepted raw vector from neighbours that share no data
with it (overlapping windows and pairs sharing a frame are left out) and reports the median
miss. Lower is better, and it cannot go below the raw measurement noise. On the sample clips,
temporal smoothing beats spatial: temporal 9–25 reaches ≈0.30–0.35 px, while any spatial
smoothing scores worse than temporal alone, because the fog field really varies from one grid
cell to the next.

## Layout

- `crates/piv-core`: algorithms with no I/O (FFT correlation, multi-pass, validation, preprocessing, synthetic images)
- `crates/piv-io`: FFmpeg decoding, capture-rate metadata, scale, HDF5/NPZ writers, quiver rendering
- `crates/piv-cli`: the `piv` binary
