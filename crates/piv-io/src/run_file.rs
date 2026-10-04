//! Reading an analysed run back from HDF5, and writing a smoothed copy of it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group, H5Type};
use ndarray::{Array1, Array3};
use serde_json::Value;

use crate::write::hdf5_out::write_attrs;

/// Everything `piv analyze` wrote to an HDF5 file.
pub struct RunData {
    pub path: PathBuf,
    /// The run metadata (`run_json`).
    pub meta: Value,
    pub x: Array1<f32>,
    pub y: Array1<f32>,
    pub frame_a: Vec<u64>,
    pub frame_b: Vec<u64>,
    pub t: Vec<f64>,
    pub dx: Array3<f32>,
    pub dy: Array3<f32>,
    pub peak_ratio: Array3<f32>,
    pub flags: Array3<u8>,
}

impl RunData {
    pub fn dt(&self) -> Result<f64> {
        self.meta["dt_s"]
            .as_f64()
            .context("run metadata has no dt_s")
    }

    pub fn m_per_px(&self) -> Option<f64> {
        self.meta["scale"]["m_per_px"].as_f64()
    }

    pub fn len(&self) -> usize {
        self.frame_a.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frame_a.is_empty()
    }
}

fn read_str_attr(file: &File, name: &str) -> Result<Option<String>> {
    if !file.attr_names()?.iter().any(|n| n == name) {
        return Ok(None);
    }
    let v: VarLenUnicode = file.attr(name)?.read_scalar()?;
    Ok(Some(v.as_str().to_string()))
}

pub fn read_run(path: &Path) -> Result<RunData> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    if read_str_attr(&file, "smoothing_json")?.is_some() {
        bail!(
            "{} is already a smoothed file; smooth the original run instead",
            path.display()
        );
    }
    let meta_json = read_str_attr(&file, "run_json")?.with_context(|| {
        format!(
            "{} has no run_json attribute; was it written by piv analyze?",
            path.display()
        )
    })?;
    let ds = |name: &str| {
        file.dataset(name)
            .with_context(|| format!("{}: missing {name}", path.display()))
    };
    Ok(RunData {
        path: path.to_path_buf(),
        meta: serde_json::from_str(&meta_json).context("parsing run_json")?,
        x: ds("grid/x")?.read_1d()?,
        y: ds("grid/y")?.read_1d()?,
        frame_a: ds("pairs/frame_a")?.read_raw()?,
        frame_b: ds("pairs/frame_b")?.read_raw()?,
        t: ds("pairs/t")?.read_raw()?,
        dx: ds("fields/dx")?.read()?,
        dy: ds("fields/dy")?.read()?,
        peak_ratio: ds("fields/peak_ratio")?.read()?,
        flags: ds("fields/flags")?.read()?,
    })
}

/// A smoothed series to write alongside the raw one.
pub struct SmoothedFields<'a> {
    pub dx: &'a Array3<f32>,
    pub dy: &'a Array3<f32>,
    pub support: &'a Array3<u16>,
    /// The smoothing settings and source, stored as `smoothing_json` and flat attributes.
    pub settings: Value,
}

fn put3<T: H5Type>(g: &Group, name: &str, data: &Array3<T>) -> Result<()> {
    let (_, ny, nx) = data.dim();
    g.new_dataset_builder()
        .with_data(data)
        .chunk((1, ny, nx))
        .shuffle()
        .deflate(4)
        .create(name)
        .with_context(|| format!("writing {name}"))?;
    Ok(())
}

/// Writes a smoothed run. The velocity datasets have the same names as in `piv analyze` output,
/// so code reading `fields/u`, `fields/flags`, … works unchanged. A smoothed value has no
/// correlation peak, so there is no `fields/peak_ratio`; the raw one is under `/raw`.
///
/// ```text
/// /fields/dx, dy, u, v, (u_mps, v_mps)   smoothed values (NaN where there is no output)
/// /fields/support                        raw vectors that contributed (temporal window, or
///                                        spatial kernel when smoothing only spatially)
/// /fields/flags                          1 = no smoothed value
/// /raw/dx, dy, flags, peak_ratio         the original fields, unchanged
/// /grid, /pairs                          copied
/// root attributes: the original run metadata plus `smoothing_json`
/// ```
pub fn write_smoothed(path: &Path, run: &RunData, sm: &SmoothedFields) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    write_attrs(&file, &run.meta, "run_json")?;
    write_attrs(&file, &sm.settings, "smoothing_json")?;

    let grid = file.create_group("grid")?;
    grid.new_dataset_builder().with_data(&run.x).create("x")?;
    grid.new_dataset_builder().with_data(&run.y).create("y")?;
    let pairs = file.create_group("pairs")?;
    pairs
        .new_dataset_builder()
        .with_data(&run.frame_a)
        .create("frame_a")?;
    pairs
        .new_dataset_builder()
        .with_data(&run.frame_b)
        .create("frame_b")?;
    pairs.new_dataset_builder().with_data(&run.t).create("t")?;

    let inv_dt = (1.0 / run.dt()?) as f32;
    let fields = file.create_group("fields")?;
    put3(&fields, "dx", sm.dx)?;
    put3(&fields, "dy", sm.dy)?;
    let u = sm.dx.mapv(|d| d * inv_dt);
    let v = sm.dy.mapv(|d| d * inv_dt);
    if let Some(k) = run.m_per_px() {
        let k = k as f32;
        put3(&fields, "u_mps", &u.mapv(|x| x * k))?;
        put3(&fields, "v_mps", &v.mapv(|x| x * k))?;
    }
    put3(&fields, "u", &u)?;
    put3(&fields, "v", &v)?;
    put3(&fields, "support", sm.support)?;
    let flags = sm.dx.mapv(|d| {
        if d.is_finite() {
            0u8
        } else {
            piv_core::flags::NO_SIGNAL
        }
    });
    put3(&fields, "flags", &flags)?;

    let raw = file.create_group("raw")?;
    put3(&raw, "dx", &run.dx)?;
    put3(&raw, "dy", &run.dy)?;
    put3(&raw, "flags", &run.flags)?;
    put3(&raw, "peak_ratio", &run.peak_ratio)?;
    file.flush()?;
    Ok(())
}
