use serde::Serialize;

use crate::PivError;

/// A regular grid of square interrogation windows laid over an image.
///
/// The grid is centred, so any leftover pixels are split evenly between the two edges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Grid {
    pub window: usize,
    pub step: usize,
    pub nx: usize,
    pub ny: usize,
    pub x0: usize,
    pub y0: usize,
}

impl Grid {
    /// `overlap` is the fraction of the window shared by neighbouring windows, in `[0, 1)`.
    pub fn new(height: usize, width: usize, window: usize, overlap: f32) -> crate::Result<Self> {
        if !(0.0..1.0).contains(&overlap) {
            return Err(PivError::Config(format!(
                "overlap {overlap} must be in [0, 1)"
            )));
        }
        if window == 0 || window > width || window > height {
            return Err(PivError::ImageTooSmall {
                height,
                width,
                window,
            });
        }
        let step = ((window as f32 * (1.0 - overlap)).round() as usize).max(1);
        let nx = (width - window) / step + 1;
        let ny = (height - window) / step + 1;
        Ok(Self {
            window,
            step,
            nx,
            ny,
            x0: (width - window - (nx - 1) * step) / 2,
            y0: (height - window - (ny - 1) * step) / 2,
        })
    }

    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Top-left pixel `(x, y)` of window `(ix, iy)`.
    pub fn window_origin(&self, ix: usize, iy: usize) -> (usize, usize) {
        (self.x0 + ix * self.step, self.y0 + iy * self.step)
    }

    /// Pixel-centre coordinate of column `ix` (pixel `i` spans `[i - 0.5, i + 0.5]`).
    pub fn center_x(&self, ix: usize) -> f32 {
        (self.x0 + ix * self.step) as f32 + (self.window as f32 - 1.0) / 2.0
    }

    pub fn center_y(&self, iy: usize) -> f32 {
        (self.y0 + iy * self.step) as f32 + (self.window as f32 - 1.0) / 2.0
    }

    pub fn xs(&self) -> Vec<f32> {
        (0..self.nx).map(|i| self.center_x(i)).collect()
    }

    pub fn ys(&self) -> Vec<f32> {
        (0..self.ny).map(|i| self.center_y(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_overlap_grid_on_full_hd() {
        let g = Grid::new(1080, 1920, 32, 0.5).unwrap();
        assert_eq!(g.step, 16);
        assert_eq!((g.nx, g.ny), (119, 66));
        // 1920 - 32 - 118*16 = 0 leftover; 1080 - 32 - 65*16 = 8 → 4 px top margin.
        assert_eq!((g.x0, g.y0), (0, 4));
        assert_eq!(g.center_x(0), 15.5);
        let (x, y) = g.window_origin(g.nx - 1, g.ny - 1);
        assert!(x + 32 <= 1920 && y + 32 <= 1080);
    }

    #[test]
    fn no_overlap_and_exact_fit() {
        let g = Grid::new(64, 64, 32, 0.0).unwrap();
        assert_eq!((g.nx, g.ny, g.step, g.x0), (2, 2, 32, 0));
        assert_eq!(g.xs(), vec![15.5, 47.5]);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(Grid::new(16, 64, 32, 0.5).is_err());
        assert!(Grid::new(64, 64, 32, 1.0).is_err());
        assert!(Grid::new(64, 64, 32, -0.1).is_err());
    }
}
