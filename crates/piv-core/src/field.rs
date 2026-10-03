use ndarray::Array2;

use crate::Grid;

/// Per-vector quality flags. A vector may carry several.
pub mod flags {
    /// Too little image signal in either window; `dx`/`dy` are NaN.
    pub const NO_SIGNAL: u8 = 1 << 0;
    /// Primary-to-secondary correlation peak ratio below threshold.
    pub const LOW_PPR: u8 = 1 << 1;
    /// Failed the normalized median test.
    pub const MEDIAN_OUTLIER: u8 = 1 << 2;
    /// Value was replaced by the median of valid neighbours.
    pub const REPLACED: u8 = 1 << 3;

    /// Vectors with any of these flags are not trustworthy measurements.
    pub const INVALID: u8 = NO_SIGNAL | LOW_PPR | MEDIAN_OUTLIER;
}

/// Displacement field for one image pair, in pixels, sampled on a [`Grid`].
///
/// Arrays are `[ny, nx]`. Vectors flagged [`flags::NO_SIGNAL`] hold NaN.
#[derive(Debug, Clone)]
pub struct VectorField {
    pub grid: Grid,
    pub dx: Array2<f32>,
    pub dy: Array2<f32>,
    pub peak_ratio: Array2<f32>,
    pub flags: Array2<u8>,
}

impl VectorField {
    pub fn new(grid: Grid) -> Self {
        let shape = (grid.ny, grid.nx);
        Self {
            grid,
            dx: Array2::from_elem(shape, f32::NAN),
            dy: Array2::from_elem(shape, f32::NAN),
            peak_ratio: Array2::zeros(shape),
            flags: Array2::from_elem(shape, flags::NO_SIGNAL),
        }
    }

    pub fn xs(&self) -> Vec<f32> {
        self.grid.xs()
    }

    pub fn ys(&self) -> Vec<f32> {
        self.grid.ys()
    }

    /// Bilinear interpolation of the displacement at pixel `(x, y)`, clamped to the grid
    /// edges. NaN samples are ignored; returns `(0, 0)` if every neighbour is NaN.
    pub fn sample(&self, x: f32, y: f32) -> (f32, f32) {
        let g = &self.grid;
        let fx = ((x - g.center_x(0)) / g.step as f32).clamp(0.0, (g.nx - 1) as f32);
        let fy = ((y - g.center_y(0)) / g.step as f32).clamp(0.0, (g.ny - 1) as f32);
        let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
        let (tx, ty) = (fx - ix as f32, fy - iy as f32);
        let (ix1, iy1) = ((ix + 1).min(g.nx - 1), (iy + 1).min(g.ny - 1));

        let mut acc = (0.0, 0.0, 0.0);
        for (r, c, w) in [
            (iy, ix, (1.0 - tx) * (1.0 - ty)),
            (iy, ix1, tx * (1.0 - ty)),
            (iy1, ix, (1.0 - tx) * ty),
            (iy1, ix1, tx * ty),
        ] {
            let (u, v) = (self.dx[[r, c]], self.dy[[r, c]]);
            if w > 0.0 && u.is_finite() && v.is_finite() {
                acc = (acc.0 + w * u, acc.1 + w * v, acc.2 + w);
            }
        }
        if acc.2 > 0.0 {
            (acc.0 / acc.2, acc.1 / acc.2)
        } else {
            (0.0, 0.0)
        }
    }

    /// Number of vectors that are real measurements (not missing, not rejected).
    pub fn valid_count(&self) -> usize {
        self.flags
            .iter()
            .filter(|&&f| f & flags::INVALID == 0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_interpolates_and_skips_nan() {
        let grid = Grid::new(64, 64, 32, 0.0).unwrap(); // centres at 15.5, 47.5
        let mut f = VectorField::new(grid);
        f.dx.assign(&ndarray::array![[0.0, 2.0], [0.0, 2.0]]);
        f.dy.fill(1.0);
        assert_eq!(f.sample(31.5, 20.0), (1.0, 1.0));
        assert_eq!(f.sample(-100.0, 15.5), (0.0, 1.0));
        f.dx[[0, 0]] = f32::NAN;
        // Halfway between a NaN and a valid sample, only the valid one contributes.
        assert_eq!(f.sample(31.5, 15.5), (2.0, 1.0));
        // Exactly on an isolated NaN with no other weight: no information.
        assert_eq!(f.sample(15.5, 15.5), (0.0, 0.0));
    }
}
