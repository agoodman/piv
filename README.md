# PIV

Planar image velocimetry for high-speed light-sheet video.

- [`rust/`](rust/README.md): the reference implementation and the `piv` command-line tool
  (analyze, smooth, HDF5/NPZ output, quiver videos).
- [`swift/`](swift/): `PIVKit`, a Swift package that runs the same analysis on the GPU with
  Metal, plus the `pivkit` command-line tool and a Mac app. The package manifest is
  [`Package.swift`](Package.swift) at the repository root.
- [`docs/`](docs/): phase plans, design notes and findings.

Sample footage goes in `clips/` and run outputs in `out/`; both are shared by the two
implementations and are not committed.
