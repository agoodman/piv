/// Per-vector quality flags. A vector may carry several. Same bit values as `piv_core::flags`.
public struct VectorFlags: OptionSet, Hashable, Sendable, Codable {
    public let rawValue: UInt8
    public init(rawValue: UInt8) { self.rawValue = rawValue }

    /// Too little image signal in either window; `dx`/`dy` are NaN.
    public static let noSignal = VectorFlags(rawValue: 1 << 0)
    /// Primary-to-secondary correlation peak ratio below threshold.
    public static let lowPPR = VectorFlags(rawValue: 1 << 1)
    /// Failed the normalized median test.
    public static let medianOutlier = VectorFlags(rawValue: 1 << 2)
    /// Value was replaced by the median of valid neighbours.
    public static let replaced = VectorFlags(rawValue: 1 << 3)
    /// Displacement larger than the first-pass window can measure.
    public static let outOfRange = VectorFlags(rawValue: 1 << 4)
    /// Vectors with any of these flags are not trustworthy measurements.
    public static let invalid: VectorFlags = [.noSignal, .lowPPR, .medianOutlier, .outOfRange]

    /// Not missing and not rejected.
    public var isValid: Bool { isDisjoint(with: .invalid) }
}

/// Displacement field for one image pair, in pixels, sampled on a ``Grid``.
///
/// Arrays are row-major `[ny][nx]` (index `iy * nx + ix`). Vectors flagged
/// ``VectorFlags/noSignal`` hold NaN.
public struct VectorField: Sendable, Equatable {
    public let grid: Grid
    public var dx: [Float]
    public var dy: [Float]
    public var peakRatio: [Float]
    public var flags: [VectorFlags]

    public init(grid: Grid) {
        self.grid = grid
        dx = Array(repeating: .nan, count: grid.count)
        dy = Array(repeating: .nan, count: grid.count)
        peakRatio = Array(repeating: 0, count: grid.count)
        flags = Array(repeating: .noSignal, count: grid.count)
    }

    public init(grid: Grid, dx: [Float], dy: [Float], peakRatio: [Float], flags: [VectorFlags]) {
        precondition([dx.count, dy.count, peakRatio.count, flags.count].allSatisfy { $0 == grid.count })
        self.grid = grid
        self.dx = dx
        self.dy = dy
        self.peakRatio = peakRatio
        self.flags = flags
    }

    @inline(__always) public func index(_ ix: Int, _ iy: Int) -> Int { iy * grid.nx + ix }

    /// Bilinear interpolation of the displacement at pixel `(x, y)`, clamped to the grid
    /// edges. NaN samples are ignored; returns `(0, 0)` if every neighbour is NaN.
    public func sample(x: Float, y: Float) -> (dx: Float, dy: Float) {
        let g = grid
        let fx = min(max((x - g.centerX(0)) / Float(g.step), 0), Float(g.nx - 1))
        let fy = min(max((y - g.centerY(0)) / Float(g.step), 0), Float(g.ny - 1))
        let (ix, iy) = (Int(fx.rounded(.down)), Int(fy.rounded(.down)))
        let (tx, ty) = (fx - Float(ix), fy - Float(iy))
        let (ix1, iy1) = (min(ix + 1, g.nx - 1), min(iy + 1, g.ny - 1))
        var (au, av, aw): (Float, Float, Float) = (0, 0, 0)
        for (r, c, w) in [
            (iy, ix, (1 - tx) * (1 - ty)),
            (iy, ix1, tx * (1 - ty)),
            (iy1, ix, (1 - tx) * ty),
            (iy1, ix1, tx * ty),
        ] {
            let k = r * g.nx + c
            let (u, v) = (dx[k], dy[k])
            if w > 0, u.isFinite, v.isFinite {
                au += w * u
                av += w * v
                aw += w
            }
        }
        return aw > 0 ? (au / aw, av / aw) : (0, 0)
    }

    /// Number of vectors that are real measurements (not missing, not rejected).
    public var validCount: Int { flags.lazy.filter(\.isValid).count }
}
