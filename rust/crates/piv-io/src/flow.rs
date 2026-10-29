//! Per-pair flow statistics for a run (mean vorticity, kinetic energy) and their units.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView2};
use piv_core::{FlowStats, flags, flow_stats};

/// Air at 20 °C, kg/m³.
pub const AIR_DENSITY: f64 = 1.204;

/// Vectors that count toward the statistics: accepted or interpolated (`REPLACED`), and finite.
pub fn usable(dx: ArrayView2<f32>, fl: ArrayView2<u8>) -> Array2<bool> {
    ndarray::Zip::from(dx)
        .and(fl)
        .map_collect(|d, &f| d.is_finite() && (f & flags::INVALID == 0 || f & flags::REPLACED != 0))
}

/// Statistics of one pair from displacements (px), in px/s units.
pub fn pair_stats(
    dx: ArrayView2<f32>,
    dy: ArrayView2<f32>,
    usable: &Array2<bool>,
    step: usize,
    dt_s: f64,
) -> FlowStats {
    let k = (1.0 / dt_s) as f32;
    flow_stats(
        &dx.mapv(|d| d * k),
        &dy.mapv(|d| d * k),
        usable,
        step as f32,
    )
}

/// The per-pair statistics of a run, in px/s units (vorticity in 1/s).
#[derive(Debug, Clone, Default)]
pub struct FlowSeries {
    /// Pair midpoint time, s.
    pub t: Vec<f64>,
    pub stats: Vec<FlowStats>,
}

impl FlowSeries {
    pub fn push(&mut self, t: f64, s: FlowStats) {
        self.t.push(t);
        self.stats.push(s);
    }

    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    pub fn column(&self, f: impl Fn(&FlowStats) -> f32) -> Vec<f32> {
        self.stats.iter().map(f).collect()
    }
}

/// Converts the px-based kinetic energy to the most physical units the run supports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyUnits {
    pub m_per_px: Option<f64>,
    pub thickness_m: Option<f64>,
    /// kg/m³
    pub density: f64,
}

impl EnergyUnits {
    /// ½⟨u² + v²⟩ (px²/s²) → (value, unit): m²/s² with a scale.
    pub fn ke_mean(&self, px2: f32) -> (f64, &'static str) {
        match self.m_per_px {
            Some(k) => (px2 as f64 * k * k, "m²/s²"),
            None => (px2 as f64, "px²/s²"),
        }
    }

    /// ½ Σ (u² + v²) · A (px⁴/s²) → (value, unit): ×ρ per metre of sheet depth (mJ/m) with a
    /// scale, × thickness (µJ) with a thickness too.
    pub fn ke_total(&self, px4: f32) -> (f64, &'static str) {
        match (self.m_per_px, self.thickness_m) {
            (Some(k), Some(t)) => (px4 as f64 * k.powi(4) * self.density * t * 1e6, "µJ"),
            (Some(k), None) => (px4 as f64 * k.powi(4) * self.density * 1e3, "mJ/m"),
            _ => (px4 as f64, "px⁴/s²"),
        }
    }

    /// Short description of what [`ke_total`](Self::ke_total) means.
    pub fn ke_total_label(&self) -> &'static str {
        match (self.m_per_px, self.thickness_m) {
            (Some(_), Some(_)) => "in-plane KE in the measured slab",
            (Some(_), None) => "in-plane KE per metre of sheet depth",
            _ => "in-plane KE per unit density and depth",
        }
    }
}

/// Writes the series to group `group` (e.g. `stats` or `raw/stats`) of an existing HDF5 file.
#[cfg(feature = "hdf5")]
pub fn write_stats_hdf5(
    path: &Path,
    group: &str,
    series: &FlowSeries,
    units: &EnergyUnits,
) -> Result<()> {
    use hdf5_metno::File;
    use hdf5_metno::types::VarLenUnicode;

    let file = File::open_rw(path).with_context(|| format!("opening {}", path.display()))?;
    let g = file
        .create_group(group)
        .with_context(|| format!("creating {group}"))?;
    for (name, values, unit) in columns(series) {
        let ds = g
            .new_dataset_builder()
            .with_data(&values)
            .create(name)
            .with_context(|| format!("writing {group}/{name}"))?;
        let unit: VarLenUnicode = unit.parse().expect("no NUL");
        ds.new_attr::<VarLenUnicode>()
            .create("units")?
            .write_scalar(&unit)?;
    }
    let note: VarLenUnicode = conversion_note(units).parse().expect("no NUL");
    g.new_attr::<VarLenUnicode>()
        .create("notes")?
        .write_scalar(&note)?;
    file.flush()?;
    Ok(())
}

/// Writes the series as `stats.npz` in an NPZ output directory.
pub fn write_stats_npz(dir: &Path, series: &FlowSeries) -> Result<()> {
    let path = dir.join("stats.npz");
    let file =
        std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut npz = ndarray_npy::NpzWriter::new_compressed(file);
    npz.add_array("t", &Array1::from(series.t.clone()))?;
    for (name, values, _) in columns(series) {
        npz.add_array(name, &values)?;
    }
    npz.finish()?;
    Ok(())
}

fn columns(s: &FlowSeries) -> Vec<(&'static str, Array1<f32>, &'static str)> {
    let col = |f: fn(&FlowStats) -> f32| Array1::from(s.column(f));
    vec![
        ("vorticity_mean", col(|x| x.vorticity_mean), "1/s"),
        ("vorticity_abs_mean", col(|x| x.vorticity_abs_mean), "1/s"),
        ("ke_mean", col(|x| x.ke_mean), "px^2/s^2"),
        ("ke_sum", col(|x| x.ke_sum), "px^4/s^2"),
        ("coverage", col(|x| x.coverage), "fraction"),
    ]
}

fn conversion_note(u: &EnergyUnits) -> String {
    let mut s = String::from(
        "vorticity: + = counter-clockwise as seen in the video (y up). \
         ke_mean = 1/2 <u^2+v^2>; ke_sum = 1/2 sum (u^2+v^2) * grid_step^2 over measured cells \
         (accepted + interpolated), in-plane only. \
         Physical: ke_mean * m_per_px^2 -> m^2/s^2; ke_sum * m_per_px^4 * density -> J per m of \
         sheet depth; * thickness -> J.",
    );
    if let Some(k) = u.m_per_px {
        s += &format!(" This run: m_per_px = {k:e}");
        if let Some(t) = u.thickness_m {
            s += &format!(", thickness = {t} m");
        }
        s += &format!(", density used for overlays = {} kg/m^3.", u.density);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_units() {
        let none = EnergyUnits {
            m_per_px: None,
            thickness_m: None,
            density: AIR_DENSITY,
        };
        assert_eq!(none.ke_total(5.0), (5.0, "px⁴/s²"));
        assert_eq!(none.ke_mean(5.0), (5.0, "px²/s²"));
        // 1 mm per px: 100 px/s = 0.1 m/s over a 10 × 10 px cell (1e-4 m²), 1 mm deep:
        // ½ ρ v² V = 0.5 × 1.204 × 0.01 × 1e-7 J = 6.02e-10 J = 6.02e-4 µJ.
        let ke_sum_px = 0.5 * 100.0f32 * 100.0 * 10.0 * 10.0;
        let full = EnergyUnits {
            m_per_px: Some(1e-3),
            thickness_m: Some(1e-3),
            density: AIR_DENSITY,
        };
        let (v, u) = full.ke_total(ke_sum_px);
        assert_eq!(u, "µJ");
        assert!((v - 6.02e-4).abs() < 1e-8, "{v}");
        let per_m = EnergyUnits {
            thickness_m: None,
            ..full
        };
        let (v, u) = per_m.ke_total(ke_sum_px);
        assert_eq!(u, "mJ/m");
        assert!((v - 6.02e-4).abs() < 1e-8, "{v}");
        let (v, u) = full.ke_mean(5000.0);
        assert_eq!(u, "m²/s²");
        assert!((v - 5e-3).abs() < 1e-12);
    }

    #[test]
    fn usable_includes_replaced() {
        let dx = ndarray::arr2(&[[1.0, 1.0, f32::NAN, 1.0]]);
        let fl = ndarray::arr2(&[[
            0,
            flags::MEDIAN_OUTLIER | flags::REPLACED,
            0,
            flags::LOW_PPR,
        ]]);
        let u = usable(dx.view(), fl.view());
        assert_eq!(u.as_slice().unwrap(), &[true, true, false, false]);
    }
}
