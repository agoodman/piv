#!/usr/bin/env python3
"""Compare a Rust run (`piv analyze --format npz` directory, or HDF5 if h5py is installed)
with a Swift run (.npy directory from `pivkit analyze`) of the same clip and settings. Only
numpy is needed for NPZ.

    python3 rust/scripts/compare_runs.py out/run_rust/ out/run_swift/

Reports, over the pairs both runs analysed:
  - flag agreement (exact, and NO_SIGNAL / rejected / replaced separately),
  - displacement differences where both runs have an accepted vector (median, p95, p99, max),
  - and the per-pair flow statistics (mean vorticity, KE) side by side.
"""
import json
import sys
from pathlib import Path

import numpy as np

NO_SIGNAL, LOW_PPR, MEDIAN, REPLACED, OUT_OF_RANGE = 1, 2, 4, 8, 16
INVALID = NO_SIGNAL | LOW_PPR | MEDIAN | OUT_OF_RANGE


def load_rust(path):
    p = Path(path)
    if p.is_dir():
        pairs = sorted(p.glob("pair_*.npz"))
        z = [np.load(q) for q in pairs]
        stats = dict(np.load(p / "stats.npz")) if (p / "stats.npz").exists() else {}
        return {
            "dx": np.stack([a["dx"] for a in z]), "dy": np.stack([a["dy"] for a in z]),
            "flags": np.stack([a["flags"] for a in z]), "frame_a": np.array([a["frames"][0] for a in z]),
            "grid_x": z[0]["x"], "grid_y": z[0]["y"], "stats": stats,
            "meta": json.loads((p / "run.json").read_text()),
        }
    import h5py
    with h5py.File(path) as f:
        return {
            "dx": f["fields/dx"][:], "dy": f["fields/dy"][:], "flags": f["fields/flags"][:],
            "frame_a": f["pairs/frame_a"][:], "grid_x": f["grid/x"][:], "grid_y": f["grid/y"][:],
            "stats": {k: f[f"stats/{k}"][:] for k in f["stats"].keys()} if "stats" in f else {},
            "meta": json.loads(f.attrs["run_json"]),
        }


def load_swift(path):
    d = Path(path)
    load = lambda n: np.load(d / f"{n}.npy")
    stats = {p.stem[len("stats_"):]: np.load(p) for p in d.glob("stats_*.npy")}
    return {
        "dx": load("dx"), "dy": load("dy"), "flags": load("flags"), "frame_a": load("frame_a"),
        "grid_x": load("grid_x"), "grid_y": load("grid_y"), "stats": stats,
        "meta": json.loads((d / "run.json").read_text()),
    }


def pct(a, q):
    return float(np.percentile(a, q)) if a.size else float("nan")


def main(rust_path, swift_path):
    r, s = load_rust(rust_path), load_swift(swift_path)
    assert np.array_equal(r["grid_x"], s["grid_x"]) and np.array_equal(r["grid_y"], s["grid_y"]), "grids differ"
    common, ri, si = np.intersect1d(r["frame_a"], s["frame_a"], return_indices=True)
    print(f"pairs: rust {len(r['frame_a'])}, swift {len(s['frame_a'])}, common {len(common)}")
    print(f"channel: rust {r['meta'].get('channel')}, swift {s['meta'].get('channel')}")
    rf, sf = r["flags"][ri], s["flags"][si]
    n = rf.size
    print(f"flags identical:            {np.mean(rf == sf) * 100:7.3f} %  ({np.sum(rf != sf)} of {n} differ)")
    for name, bit in [("no signal", NO_SIGNAL), ("low peak ratio", LOW_PPR), ("median outlier", MEDIAN),
                      ("replaced", REPLACED), ("out of range", OUT_OF_RANGE)]:
        a, b = (rf & bit) != 0, (sf & bit) != 0
        print(f"  {name:<16} rust {a.mean() * 100:6.2f} %  swift {b.mean() * 100:6.2f} %  disagree {np.mean(a != b) * 100:6.3f} %")

    both = ((rf & INVALID) == 0) & ((sf & INVALID) == 0)
    d = np.hypot(r["dx"][ri] - s["dx"][si], r["dy"][ri] - s["dy"][si])[both]
    print(f"accepted by both: {both.mean() * 100:.2f} % of vectors")
    print(f"|Δd| (px) over those: median {pct(d, 50):.2e}  p95 {pct(d, 95):.2e}  p99 {pct(d, 99):.2e}  max {d.max():.2e}")
    for t in [1e-4, 1e-3, 1e-2, 0.1]:
        print(f"  fraction with |Δd| > {t:g} px: {np.mean(d > t) * 100:.3f} %")

    for k in ["vorticity_mean", "ke_mean", "coverage"]:
        if k in r["stats"] and k in s["stats"]:
            a, b = r["stats"][k][ri], s["stats"][k][si]
            ok = np.isfinite(a) & np.isfinite(b)
            rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), 1e-12)
            print(f"stats {k:<16} median rel. diff {pct(rel, 50):.2e}, max {rel.max():.2e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
