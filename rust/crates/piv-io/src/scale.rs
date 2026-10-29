//! Approximate physical scale (metres per pixel) from user-entered optics.
//!
//! This is not a calibration: it assumes a distortion-free pinhole camera looking straight at
//! the light sheet. A measured `--px-per-mm` from a ruler or target in the sheet plane is
//! always preferable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FovAxis {
    Diagonal,
    Horizontal,
    Vertical,
}

/// What the user told us about the optics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScaleSpec {
    pub px_per_mm: Option<f64>,
    pub fov_deg: Option<f64>,
    pub fov_axis: Option<FovAxis>,
    pub sheet_distance_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Scale {
    pub m_per_px: f64,
    pub method: String,
}

impl ScaleSpec {
    /// Resolves a scale for a full frame of `width × height` pixels. Returns `None` when the
    /// inputs are insufficient (e.g. a field of view without a distance).
    pub fn resolve(&self, width: usize, height: usize) -> Option<Scale> {
        if let Some(s) = self.px_per_mm {
            return Some(Scale {
                m_per_px: 1e-3 / s,
                method: format!("--px-per-mm {s}"),
            });
        }
        let (fov, d) = (self.fov_deg?, self.sheet_distance_m?);
        let axis = self.fov_axis.unwrap_or(FovAxis::Diagonal);
        let (w, h) = (width as f64, height as f64);
        let span_px = match axis {
            FovAxis::Diagonal => w.hypot(h),
            FovAxis::Horizontal => w,
            FovAxis::Vertical => h,
        };
        let span_m = 2.0 * d * (fov.to_radians() / 2.0).tan();
        Some(Scale {
            m_per_px: span_m / span_px,
            method: format!("pinhole estimate: {fov}° {axis:?} FOV at {d} m over {span_px:.1} px")
                .to_lowercase(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fov(axis: FovAxis, d: Option<f64>) -> ScaleSpec {
        ScaleSpec {
            fov_deg: Some(84.0),
            fov_axis: Some(axis),
            sheet_distance_m: d,
            ..Default::default()
        }
    }

    #[test]
    fn c0141_working_estimate() {
        // 84° diagonal at 300 mm over a 1920×1080 frame ≈ 0.2452 mm/px (see docs/phase1-plan.md).
        let s = fov(FovAxis::Diagonal, Some(0.3))
            .resolve(1920, 1080)
            .unwrap();
        assert!((s.m_per_px * 1e3 - 0.245_240_7).abs() < 1e-6, "{s:?}");
    }

    #[test]
    fn axes_match_hand_computation() {
        let t = (42f64).to_radians().tan();
        let h = fov(FovAxis::Horizontal, Some(1.0))
            .resolve(1920, 1080)
            .unwrap();
        assert!((h.m_per_px - 2.0 * t / 1920.0).abs() < 1e-12);
        let v = fov(FovAxis::Vertical, Some(1.0))
            .resolve(1920, 1080)
            .unwrap();
        assert!((v.m_per_px - 2.0 * t / 1080.0).abs() < 1e-12);
    }

    #[test]
    fn px_per_mm_overrides_and_fov_alone_is_not_enough() {
        let mut s = fov(FovAxis::Diagonal, Some(0.3));
        s.px_per_mm = Some(4.0);
        assert_eq!(s.resolve(1920, 1080).unwrap().m_per_px, 0.25e-3);
        assert!(fov(FovAxis::Diagonal, None).resolve(1920, 1080).is_none());
        assert!(ScaleSpec::default().resolve(1920, 1080).is_none());
    }
}
