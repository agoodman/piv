//! Vector-field output: one HDF5 file per run (default) or one NPZ per pair.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use piv_core::{PivConfig, PreprocessConfig, VectorField};
use serde::{Deserialize, Serialize};

use crate::decode::{Channel, Roi};
use crate::metadata::RateSource;
use crate::scale::{Scale, ScaleSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Hdf5,
    Npz,
}

/// Everything needed to reproduce and interpret a run. Stored as HDF5 root attributes (and
/// as `run_json`), or as `run.json` next to NPZ output.
#[derive(Debug, Clone, Serialize)]
pub struct RunMetadata {
    pub tool_version: String,
    pub source_path: String,
    pub source_size: u64,
    pub camera: Option<String>,
    pub frame_width: usize,
    pub frame_height: usize,
    pub roi: Option<Roi>,
    pub channel: Channel,
    pub capture_fps: f64,
    pub capture_fps_rational: String,
    pub capture_fps_source: RateSource,
    pub capture_fps_detail: String,
    pub container_fps: f64,
    pub start_frame: usize,
    pub end_frame: usize,
    pub stride: usize,
    pub pair_step: usize,
    pub dt_s: f64,
    pub background: Option<String>,
    pub preprocess: PreprocessConfig,
    pub piv: PivConfig,
    pub scale_spec: ScaleSpec,
    pub scale: Option<Scale>,
    pub sheet_thickness_mm: Option<f64>,
}

/// One analysed frame pair.
pub struct PairRecord<'a> {
    pub frame_a: usize,
    pub frame_b: usize,
    /// Capture time of the pair midpoint, seconds from frame 0.
    pub time_s: f64,
    pub field: &'a VectorField,
}

pub trait FieldWriter {
    fn write(&mut self, rec: &PairRecord) -> Result<()>;
    fn finish(self: Box<Self>) -> Result<()>;
}

/// Grid coordinates in full-frame pixels (the ROI offset added back).
pub fn grid_coords(field: &VectorField, roi: Option<Roi>) -> (Array1<f32>, Array1<f32>) {
    let (ox, oy) = roi.map_or((0.0, 0.0), |r| (r.x as f32, r.y as f32));
    (
        field.xs().into_iter().map(|x| x + ox).collect(),
        field.ys().into_iter().map(|y| y + oy).collect(),
    )
}

/// Derived arrays shared by both writers.
struct Derived {
    u: Array2<f32>,
    v: Array2<f32>,
    physical: Option<(Array2<f32>, Array2<f32>)>,
}

fn derive(field: &VectorField, meta: &RunMetadata) -> Derived {
    let inv_dt = (1.0 / meta.dt_s) as f32;
    let u = field.dx.mapv(|d| d * inv_dt);
    let v = field.dy.mapv(|d| d * inv_dt);
    let physical = meta.scale.as_ref().map(|s| {
        let k = s.m_per_px as f32;
        (u.mapv(|x| x * k), v.mapv(|x| x * k))
    });
    Derived { u, v, physical }
}

pub fn create_writer(
    format: OutputFormat,
    path: &Path,
    meta: &RunMetadata,
    first: &VectorField,
) -> Result<Box<dyn FieldWriter>> {
    match format {
        OutputFormat::Npz => Ok(Box::new(NpzWriter::create(path, meta)?)),
        #[cfg(feature = "hdf5")]
        OutputFormat::Hdf5 => Ok(Box::new(hdf5_out::Hdf5Writer::create(path, meta, first)?)),
        #[cfg(not(feature = "hdf5"))]
        OutputFormat::Hdf5 => {
            let _ = first;
            anyhow::bail!(
                "this build has no HDF5 support; rebuild with the `hdf5` feature or use --format npz"
            )
        }
    }
}

/// One `pair_AAAAAA_BBBBBB.npz` per pair plus `run.json`, inside directory `path`.
pub struct NpzWriter {
    dir: PathBuf,
    meta: RunMetadata,
}

impl NpzWriter {
    pub fn create(dir: &Path, meta: &RunMetadata) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let json = serde_json::to_string_pretty(meta)?;
        std::fs::write(dir.join("run.json"), json)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            meta: meta.clone(),
        })
    }
}

impl FieldWriter for NpzWriter {
    fn write(&mut self, rec: &PairRecord) -> Result<()> {
        let f = rec.field;
        let d = derive(f, &self.meta);
        let (x, y) = grid_coords(f, self.meta.roi);
        let path = self
            .dir
            .join(format!("pair_{:06}_{:06}.npz", rec.frame_a, rec.frame_b));
        let file =
            std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
        let mut npz = ndarray_npy::NpzWriter::new_compressed(file);
        npz.add_array("x", &x)?;
        npz.add_array("y", &y)?;
        npz.add_array("dx", &f.dx)?;
        npz.add_array("dy", &f.dy)?;
        npz.add_array("u", &d.u)?;
        npz.add_array("v", &d.v)?;
        if let Some((um, vm)) = &d.physical {
            npz.add_array("u_mps", um)?;
            npz.add_array("v_mps", vm)?;
        }
        npz.add_array("peak_ratio", &f.peak_ratio)?;
        npz.add_array("flags", &f.flags)?;
        npz.add_array(
            "frames",
            &Array1::from(vec![rec.frame_a as u64, rec.frame_b as u64]),
        )?;
        npz.add_array("t", &ndarray::arr0(rec.time_s))?;
        npz.finish()?;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "hdf5")]
pub mod hdf5_out {
    use super::*;
    use hdf5_metno::types::VarLenUnicode;
    use hdf5_metno::{Dataset, Extent, Extents, File, Group, H5Type, Location};
    use serde_json::Value;

    /// Layout (arrays `[n_pairs, ny, nx]`, chunked one pair per chunk):
    ///
    /// ```text
    /// /grid/x, /grid/y                      window centres, full-frame px
    /// /fields/dx, /fields/dy                displacement, px
    /// /fields/u, /fields/v                  velocity, px/s
    /// /fields/u_mps, /fields/v_mps          velocity, m/s (only with a scale)
    /// /fields/peak_ratio, /fields/flags
    /// /pairs/frame_a, /pairs/frame_b, /pairs/t
    /// root attributes: run metadata, plus `run_json`
    /// ```
    pub struct Hdf5Writer {
        file: File,
        meta: RunMetadata,
        n: usize,
        shape: (usize, usize),
        f32_sets: Vec<(&'static str, Dataset)>,
        flags: Dataset,
        frame_a: Dataset,
        frame_b: Dataset,
        t: Dataset,
    }

    fn stack<T: H5Type>(g: &Group, name: &str, (ny, nx): (usize, usize)) -> Result<Dataset> {
        g.new_dataset::<T>()
            .shape(Extents::new(vec![
                Extent::resizable(0),
                Extent::fixed(ny),
                Extent::fixed(nx),
            ]))
            .chunk((1, ny, nx))
            .shuffle()
            .deflate(4)
            .create(name)
            .with_context(|| format!("creating dataset {name}"))
    }

    fn series<T: H5Type>(g: &Group, name: &str) -> Result<Dataset> {
        g.new_dataset::<T>()
            .shape(Extents::new(vec![Extent::resizable(0)]))
            .chunk(1024)
            .create(name)
            .with_context(|| format!("creating dataset {name}"))
    }

    /// Writes `json` as a pretty-printed string attribute `json_name`, plus each top-level
    /// field as its own attribute (numbers, strings and bools natively; nested values as JSON).
    pub fn write_attrs(loc: &Location, json: &Value, json_name: &str) -> Result<()> {
        let str_attr = |name: &str, s: &str| -> Result<()> {
            let v: VarLenUnicode = s
                .parse()
                .map_err(|e| anyhow::anyhow!("attribute {name}: {e:?}"))?;
            loc.new_attr::<VarLenUnicode>()
                .create(name)?
                .write_scalar(&v)?;
            Ok(())
        };
        str_attr(json_name, &serde_json::to_string_pretty(json)?)?;
        for (k, v) in json.as_object().expect("struct serializes to an object") {
            match v {
                Value::Null => {}
                Value::Bool(b) => loc.new_attr::<bool>().create(k.as_str())?.write_scalar(b)?,
                Value::Number(n) => match (n.as_u64(), n.as_f64()) {
                    (Some(u), _) => loc.new_attr::<u64>().create(k.as_str())?.write_scalar(&u)?,
                    (None, Some(f)) => {
                        loc.new_attr::<f64>().create(k.as_str())?.write_scalar(&f)?
                    }
                    _ => {}
                },
                Value::String(s) => str_attr(k, s)?,
                other => str_attr(k, &other.to_string())?,
            }
        }
        Ok(())
    }

    impl Hdf5Writer {
        pub fn create(path: &Path, meta: &RunMetadata, first: &VectorField) -> Result<Self> {
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            let file =
                File::create(path).with_context(|| format!("creating {}", path.display()))?;
            write_attrs(&file, &serde_json::to_value(meta)?, "run_json")?;

            let shape = first.dx.dim();
            let grid = file.create_group("grid")?;
            let (x, y) = grid_coords(first, meta.roi);
            grid.new_dataset_builder().with_data(&x).create("x")?;
            grid.new_dataset_builder().with_data(&y).create("y")?;

            let fields = file.create_group("fields")?;
            let mut names = vec!["dx", "dy", "u", "v", "peak_ratio"];
            if meta.scale.is_some() {
                names.extend(["u_mps", "v_mps"]);
            }
            let f32_sets = names
                .into_iter()
                .map(|n| Ok((n, stack::<f32>(&fields, n, shape)?)))
                .collect::<Result<_>>()?;
            let flags = stack::<u8>(&fields, "flags", shape)?;
            flags
                .new_attr::<VarLenUnicode>()
                .create("bits")?
                .write_scalar(
                    &"1=NO_SIGNAL 2=LOW_PPR 4=MEDIAN_OUTLIER 8=REPLACED 16=OUT_OF_RANGE"
                        .parse::<VarLenUnicode>()
                        .unwrap(),
                )?;

            let pairs = file.create_group("pairs")?;
            Ok(Self {
                frame_a: series::<u64>(&pairs, "frame_a")?,
                frame_b: series::<u64>(&pairs, "frame_b")?,
                t: series::<f64>(&pairs, "t")?,
                file,
                meta: meta.clone(),
                n: 0,
                shape,
                f32_sets,
                flags,
            })
        }
    }

    impl FieldWriter for Hdf5Writer {
        fn write(&mut self, rec: &PairRecord) -> Result<()> {
            let f = rec.field;
            anyhow::ensure!(f.dx.dim() == self.shape, "grid changed between pairs");
            let d = derive(f, &self.meta);
            let (n, (ny, nx)) = (self.n, self.shape);
            for (name, ds) in &self.f32_sets {
                let arr = match *name {
                    "dx" => &f.dx,
                    "dy" => &f.dy,
                    "u" => &d.u,
                    "v" => &d.v,
                    "peak_ratio" => &f.peak_ratio,
                    "u_mps" => &d.physical.as_ref().expect("scale set").0,
                    "v_mps" => &d.physical.as_ref().expect("scale set").1,
                    _ => unreachable!(),
                };
                ds.resize((n + 1, ny, nx))?;
                ds.write_slice(arr, (n, .., ..))?;
            }
            self.flags.resize((n + 1, ny, nx))?;
            self.flags.write_slice(&f.flags, (n, .., ..))?;
            for (ds, v) in [
                (&self.frame_a, rec.frame_a as u64),
                (&self.frame_b, rec.frame_b as u64),
            ] {
                ds.resize(n + 1)?;
                ds.write_slice(&[v], n..n + 1)?;
            }
            self.t.resize(n + 1)?;
            self.t.write_slice(&[rec.time_s], n..n + 1)?;
            self.n += 1;
            Ok(())
        }

        fn finish(self: Box<Self>) -> Result<()> {
            self.file.flush()?;
            Ok(())
        }
    }
}
