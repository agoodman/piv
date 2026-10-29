/// A regular grid of square interrogation windows laid over an image. Port of `piv_core::Grid`.
///
/// The grid is centred, so any leftover pixels are split evenly between the two edges.
public struct Grid: Hashable, Sendable, Codable {
    public let window: Int
    public let step: Int
    public let nx: Int
    public let ny: Int
    public let x0: Int
    public let y0: Int

    /// `overlap` is the fraction of the window shared by neighbouring windows, in `[0, 1)`.
    public init(height: Int, width: Int, window: Int, overlap: Float) throws(PIVError) {
        guard overlap >= 0, overlap < 1 else {
            throw .config("overlap \(overlap) must be in [0, 1)")
        }
        guard window > 0, window <= width, window <= height else {
            throw .imageTooSmall(height: height, width: width, window: window)
        }
        let step = max(Int((Float(window) * (1 - overlap)).rounded(.toNearestOrAwayFromZero)), 1)
        let nx = (width - window) / step + 1
        let ny = (height - window) / step + 1
        self.window = window
        self.step = step
        self.nx = nx
        self.ny = ny
        self.x0 = (width - window - (nx - 1) * step) / 2
        self.y0 = (height - window - (ny - 1) * step) / 2
    }

    public var count: Int { nx * ny }

    /// Top-left pixel `(x, y)` of window `(ix, iy)`.
    public func windowOrigin(_ ix: Int, _ iy: Int) -> (x: Int, y: Int) {
        (x0 + ix * step, y0 + iy * step)
    }

    /// Pixel-centre coordinate of column `ix` (pixel `i` spans `[i - 0.5, i + 0.5]`).
    public func centerX(_ ix: Int) -> Float { Float(x0 + ix * step) + (Float(window) - 1) / 2 }
    public func centerY(_ iy: Int) -> Float { Float(y0 + iy * step) + (Float(window) - 1) / 2 }

    public var xs: [Float] { (0..<nx).map(centerX) }
    public var ys: [Float] { (0..<ny).map(centerY) }
}
