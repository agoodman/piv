use std::fmt;
use std::str::FromStr;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::PivError;

/// A grayscale image with intensities normalized to `0..=1`, indexed `[row (y), column (x)]`.
pub type Frame = Array2<f32>;

/// An exact frame rate, stored as a reduced rational so that rates like 240000/1001 survive
/// round trips without drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRate {
    num: u64,
    den: u64,
}

impl FrameRate {
    pub fn new(num: u64, den: u64) -> crate::Result<Self> {
        if num == 0 || den == 0 {
            return Err(PivError::FrameRate(format!("{num}/{den}")));
        }
        let g = gcd(num, den);
        Ok(Self {
            num: num / g,
            den: den / g,
        })
    }

    pub fn num(&self) -> u64 {
        self.num
    }

    pub fn den(&self) -> u64 {
        self.den
    }

    pub fn fps(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Seconds between frame `i` and frame `i + stride`.
    pub fn dt(&self, stride: u64) -> f64 {
        stride as f64 * self.den as f64 / self.num as f64
    }

    /// Time of (possibly fractional) frame `index`, in seconds from frame 0.
    pub fn time_of(&self, index: f64) -> f64 {
        index * self.den as f64 / self.num as f64
    }
}

impl FromStr for FrameRate {
    type Err = PivError;

    /// Accepts `240`, `239.76`, `240000/1001`, and camera-style strings such as `1000p`,
    /// `59.94i` or `240fps`. Decimal values within 0.006 of an NTSC rate (`n·1000/1001`) are
    /// snapped to that exact rational.
    fn from_str(s: &str) -> crate::Result<Self> {
        let err = || PivError::FrameRate(s.to_string());
        let t = s.trim().to_ascii_lowercase();
        let t = t
            .strip_suffix("fps")
            .or_else(|| t.strip_suffix('p'))
            .or_else(|| t.strip_suffix('i'))
            .unwrap_or(&t)
            .trim();

        if let Some((n, d)) = t.split_once('/') {
            let n = n.trim().parse().map_err(|_| err())?;
            let d = d.trim().parse().map_err(|_| err())?;
            return Self::new(n, d).map_err(|_| err());
        }
        if let Some((int, frac)) = t.split_once('.') {
            let f: f64 = t.parse().map_err(|_| err())?;
            if !f.is_finite()
                || f <= 0.0
                || frac.is_empty()
                || !frac.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(err());
            }
            let n = (f * 1001.0 / 1000.0).round();
            if n >= 1.0 && (n * 1000.0 / 1001.0 - f).abs() < 0.006 {
                return Self::new(n as u64 * 1000, 1001);
            }
            let den = 10u64.checked_pow(frac.len() as u32).ok_or_else(err)?;
            let num = format!("{int}{frac}").parse().map_err(|_| err())?;
            return Self::new(num, den).map_err(|_| err());
        }
        Self::new(t.parse().map_err(|_| err())?, 1).map_err(|_| err())
    }
}

impl fmt::Display for FrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{} fps", self.num)
        } else {
            write!(f, "{}/{} ({:.3}) fps", self.num, self.den, self.fps())
        }
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rate(s: &str) -> FrameRate {
        s.parse().unwrap()
    }

    #[test]
    fn parses_integer_and_camera_strings() {
        assert_eq!(rate("240"), FrameRate::new(240, 1).unwrap());
        assert_eq!(rate("1000p"), FrameRate::new(1000, 1).unwrap());
        assert_eq!(rate("240p"), FrameRate::new(240, 1).unwrap());
        assert_eq!(rate(" 50P "), FrameRate::new(50, 1).unwrap());
        assert_eq!(rate("120fps"), FrameRate::new(120, 1).unwrap());
    }

    #[test]
    fn parses_rationals_and_reduces() {
        assert_eq!(rate("240000/1001"), FrameRate::new(240000, 1001).unwrap());
        assert_eq!(rate("50000/1000"), FrameRate::new(50, 1).unwrap());
    }

    #[test]
    fn snaps_ntsc_decimals() {
        assert_eq!(rate("239.76"), FrameRate::new(240000, 1001).unwrap());
        assert_eq!(rate("29.97"), FrameRate::new(30000, 1001).unwrap());
        assert_eq!(rate("23.98"), FrameRate::new(24000, 1001).unwrap());
        assert_eq!(rate("119.88p"), FrameRate::new(120000, 1001).unwrap());
    }

    #[test]
    fn keeps_other_decimals_exact() {
        assert_eq!(rate("12.5"), FrameRate::new(25, 2).unwrap());
        assert_eq!(rate("1000.0"), FrameRate::new(1000, 1).unwrap());
    }

    #[test]
    fn rejects_garbage() {
        for s in ["", "0", "abc", "-5", "1/0", "0/7", "1.", "nan"] {
            assert!(s.parse::<FrameRate>().is_err(), "{s:?} should fail");
        }
    }

    #[test]
    fn dt_scales_with_stride_and_rate() {
        assert!((rate("1000").dt(1) - 1e-3).abs() < 1e-15);
        assert!((rate("1000").dt(3) - 3e-3).abs() < 1e-15);
        assert!((rate("240").dt(1) - 1.0 / 240.0).abs() < 1e-15);
        assert!((rate("239.76").dt(2) - 2.0 * 1001.0 / 240000.0).abs() < 1e-15);
        assert!((rate("50").time_of(3720.0) - 74.4).abs() < 1e-9);
    }
}
