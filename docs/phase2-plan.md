# Phase 2 plan: vorticity and kinetic energy overlays on quiver video

## 1. Context

The quiver videos from `piv analyze --video` and `piv smooth --video` show the vector field but
give no sense of how much the flow is rotating or how energetic it is, or how that changes over
the clip. Phase 2 adds two flow-wide quantities, **mean vorticity** and **total kinetic energy**,
computed for every pair. They are drawn on each video frame as:

1. **Text readouts** (top-left corner) with the current value of each quantity, its units, and
   the pair time.
2. **A chart panel** (bottom-right corner) with both quantities plotted over the whole run and a
   cursor marking the current pair.

No change to correlation, validation, or smoothing. The new code is a statistics module, an
overlay renderer, a step that writes the per-pair series to the run file, and CLI flags.

## 2. Definitions

All quantities are computed on the grid of accepted vectors (`flags & INVALID == 0`, including
`REPLACED` vectors, since the quiver draws those too; D4). Velocities are in px/s
from `fields/u, v`, or m/s when the run has a scale.

### 2.1 Vorticity

ω = ∂v/∂x − ∂u/∂y. It uses central differences on the grid (spacing = `grid.step` px) and is
defined only where all four neighbours are finite. One-sided differences are used at the edges
and next to holes when the other neighbour is valid.

- **Units are 1/s with or without a physical scale.** px/s ÷ px = 1/s, assuming square pixels,
  so the number means the same thing on runs with and without a scale.
- **Sign convention:** image y points down, so a positive ω computed directly from image axes is
  *clockwise* on screen. We flip the sign so that **positive = counter-clockwise as seen in the
  video** (the usual maths convention with y up), and say so in the legend.
- **Which "average":** the signed mean of ω over a region equals the circulation around its
  boundary divided by its area. A field holding a pair of opposite-signed eddies therefore
  averages to almost zero even when it is full of rotation. The plan reports two numbers:
  - **mean ω** (signed): overall net rotation, i.e. which way the flow turns as a whole.
  - **mean |ω|**: how much rotation there is in any direction (a measure of activity).

  Both appear in the text overlay. The chart plots the signed mean (D1).

### 2.2 Kinetic energy

Without a density and a depth we can't compute joules from the vector field alone, so the values
are layered by what the run knows:

The total is a sum over the **measured cells only** (D2): accepted and interpolated vectors,
each standing for one grid cell of area A_cell = step². The readout also shows the mean per unit
mass, ½⟨u² + v²⟩.

| Run has | Total KE shown | Units |
|---|---|---|
| no scale | ½ Σ (u² + v²) · A_cell (per unit density and sheet depth) | px⁴/s² |
| scale (`--px-per-mm` or FOV + distance) | ½ ρ Σ (u² + v²) · A_cell (per metre of sheet depth) | mJ/m |
| scale + `--sheet-thickness-mm` | **total KE in the measured slab**, ½ ρ Σ (u² + v²) · A_cell · thickness | µJ |

ρ defaults to air at 20 °C (1.204 kg/m³), and there's a new `--density` flag to override it. Only
the in-plane components are counted. The out-of-plane velocity w is unknown, so the value is a
lower bound, and the legend says "in-plane".

**Coverage caveat:** a sum over measured cells rises and falls with the number of seeded
windows as well as with the flow. When fog drifts out of the sheet the total drops even if the
air keeps moving. The coverage fraction is shown next to the total so these dips can be told
apart from real changes in the flow.

### 2.3 Noise floor (expect this in the numbers)

Raw vectors on the sample clips have ≈0.3 px of noise (CV error, phase-1 README). At stride 5 and
1000 fps (dt = 5 ms) that is σ_u ≈ 60 px/s. With the default 32 px grid step:

- **Per-point ω noise** ≈ σ_u / step ≈ 2 s⁻¹. Signed mean ω averages this away over hundreds of
  vectors. Mean |ω| does not: pure noise alone gives about 1.5 s⁻¹. The run summary prints both
  floors (the signed mean's is σ_ω/√n) so real rotation can be told from noise.
- **KE bias:** noise adds about σ_u² ≈ 3600 px²/s² to ½⟨u² + v²⟩. That matters only for slow
  flows.

Smoothed runs (`piv smooth --temporal 9`) have much less noise, and the overlay works on them
unchanged. I'll report the measured floors on C0141 and C0005, raw and t9, as part of the
results.

## 3. Where it is computed and drawn

### 3.1 Statistics: `piv-core::flowstats` (new, no I/O)

```rust
pub struct FlowStats {
    pub vorticity_mean: f32,      // 1/s, + = counter-clockwise on screen
    pub vorticity_abs_mean: f32,  // 1/s
    pub ke_mean: f32,             // ½⟨u²+v²⟩ in the input's units²
    pub coverage: f32,            // accepted / grid cells
    pub n_vorticity: usize,       // points where ω was defined
}
pub fn vorticity(u: &Array2<f32>, v: &Array2<f32>, usable: &Array2<bool>, step: f32) -> Array2<f32>;
pub fn flow_stats(u: &Array2<f32>, v: &Array2<f32>, usable: &Array2<bool>, step: f32) -> FlowStats;
```

The scale, density, and thickness conversions (§2.2) go in `piv-io` next to `Scale`. The core
module only sees arrays and a step.

### 3.2 Overlay rendering: `piv-io::overlay` (new)

- `draw_readout(img, &Readout)`: a semi-transparent dark box with 3–4 lines of text.
- `draw_chart(img, &Series, cursor: usize, &ChartStyle)`: a semi-transparent panel holding two
  stacked strips that share a time axis (signed mean vorticity on top with a zero line, total KE
  below). Each strip has its own y range. Axis limits are fixed from the whole run so the chart
  doesn't rescale between frames. The full trace is drawn dim and the part up to the cursor
  bright, with a vertical cursor line (D3).
- Panel size scales with the frame: about 35 % of the width and 22 % of the height, clamped to a
  minimum readable size. If the frame (or ROI) is narrower than ~480 px, the chart goes in a
  strip **below** the image instead of on top of it.
- Text uses `imageproc::drawing::draw_text_mut` (ab_glyph) with an embedded TTF. Plan: DejaVu
  Sans Mono (permissive Bitstream Vera licence, ~340 KB, included with `include_bytes!`, licence
  file in `crates/piv-io/assets/`) (D5).
- Colours avoid the quiver colour map (light blue → yellow → red) and the magenta used for
  replaced vectors. Vorticity is drawn in white/light grey, KE in green-cyan, and the text is
  white on ~60 % black.

### 3.3 `piv analyze --video`

Analysis is streamed, so the whole-run series (needed for fixed chart axes) only exists at the
end. Plan:

1. As now, render the plain quiver PNGs during the loop, and also record `FlowStats` for each
   pair (cheap: one pass over a 60×33 grid).
2. After the loop, before `encode_video`, a **decorate pass**: read each PNG, draw the readout
   and chart, and write it back, in parallel with rayon. On C0141 (~1 000 PNGs at 1080p) I
   expect this to take a few seconds.

So `<output>_quiver/` holds the decorated PNGs. With `--no-overlay` the step is skipped and the
output is the same as phase 1.

### 3.4 `piv smooth --video`

The fields are all in memory before rendering, so the series are computed up front and drawn
directly, without a decorate pass. The video is side-by-side raw | smoothed:

- Each panel gets its **own readout** (raw values on the left, smoothed on the right).
- **One chart**, on the right panel, plots smoothed (bright) and raw (thin, dim) traces together,
  so the effect of smoothing on the statistics shows directly.

### 3.5 Writing the series to the run file

The overlay data is useful on its own, for example for comparing runs or plotting in Python, so
it is also saved (D6):

| Dataset | Shape | Units |
|---|---|---|
| `stats/vorticity_mean`, `stats/vorticity_abs_mean` | `[pairs]` | 1/s |
| `stats/ke_mean` | `[pairs]` | px²/s² (½⟨u² + v²⟩) |
| `stats/ke_sum` | `[pairs]` | px⁴/s² (½ Σ (u² + v²) · A_cell, measured cells) |
| `stats/coverage` | `[pairs]` | fraction of grid cells measured |

Physical totals (mJ/m, µJ) depend on `--density`, so they are not stored. They are `ke_sum` ×
ρ × m_per_px⁴ (× thickness), and the conversion is in the dataset's attributes.

`piv analyze` adds these once the run finishes (HDF5 only; for NPZ, a `stats.npz` in the
directory). `piv smooth` writes them for the smoothed fields and copies the raw ones to
`raw/stats/…`. The per-point vorticity field is **not** stored unless you ask for it.

## 4. CLI changes

| Command | Option | Default | Meaning |
|---|---|---|---|
| analyze, smooth | `--no-overlay` | off | plain quiver video, as in phase 1 |
| analyze | `--density` | 1.204 | fluid density (kg/m³) for total KE |
| smooth | `--density` | from run, else 1.204 | |

`piv analyze` also prints the run-mean and peak of each quantity, plus the |ω| noise floor, in
its summary.

## 5. Verification

- **Unit tests (`flowstats`)** on analytic fields on a grid:
  - Solid-body rotation u = −Ω(y−y₀), v = Ω(x−x₀): ω = 2Ω everywhere, checking sign for
    counter-clockwise on screen with y down.
  - Uniform translation: ω = 0, KE = ½|U|².
  - Pure shear u = γy: ω = −γ (in the flipped convention).
  - Point vortex pair: signed mean ≈ 0, mean |ω| > 0.
  - Holes and edges: NaN neighbours handled, `n_vorticity` correct.
- **Synthetic end-to-end:** `piv-core::synthetic` already makes displaced particle images. Add a
  rotating field, run `analyze`, and check recovered mean ω against 2Ω (within a tolerance set
  by the measured noise).
- **Unit conversions:** px → m/s → µJ against a hand calculation.
- **Overlay:** pixel tests check that the readout and chart regions are drawn and that pixels
  outside them are unchanged (vectors are not hidden), and that the narrow-frame layout puts the
  chart below. Smoke test: `--video` on a short clip produces an MP4 of the expected size.
- **Visual review (yours):** regenerate quiver/compare videos for C0141 and C0005 (raw and t9),
  with a still frame of each for quick review. I'll report the noise floors next to the typical
  values.

## 6. Milestones (commits on `phase-2`)

1. `flowstats` in piv-core with unit tests; per-pair stats and noise floor in the `analyze`
   summary.
2. Writing `stats/` datasets (analyze, smooth, run_file reader).
3. Overlay renderer (font, readout, chart) with pixel tests.
4. Hooking it into `analyze --video` (decorate pass) and `smooth --video`; CLI flags.
5. README update, sample videos regenerated, findings added to this doc.

I'll stop after milestone 4 so you can review the videos before the docs and findings are
written.

## 7. Decisions from review (2026-10-04)

| # | Question | Decision |
|---|---|---|
| D1 | Which vorticity to chart | **Signed mean** (the readout shows signed and \|ω\|) |
| D2 | Total KE: extrapolate to the ROI, or sum measured cells | **Measured cells only**, with coverage shown beside it |
| D3 | Chart style | **Whole run with a moving cursor** |
| D4 | Include interpolated (`REPLACED`) vectors | **Yes**, in vorticity and KE |
| D5 | Font | **Embedded TTF** (DejaVu Sans Mono 2.37) |
| D6 | Save the per-pair values | **Yes**, `stats/` datasets |
| D7 | Chart placement | **Overlay** (translucent, bottom-right); strip below only for frames narrower than ~480 px |

## 8. Out of scope

- Per-point vorticity heat-map overlay (a natural next step: colour the background by ω).
- Enstrophy, divergence (a quality check on how two-dimensional the flow is), and circulation
  around user-chosen loops.
- A standalone `piv render` command to re-render videos from an `.h5` without re-analysing.
  `smooth --video` already re-decodes the source, so this would be a small step later.
