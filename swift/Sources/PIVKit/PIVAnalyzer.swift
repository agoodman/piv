import Foundation
import Metal

/// Multi-pass interrogation settings. Port of `piv_core::PivConfig`; the defaults match.
public struct PIVConfig: Hashable, Sendable, Codable {
    /// Window size for each pass, largest first. Each must be even and 2^a·3^b (GPU FFT
    /// sizes). The default `[96, 64]` suits smooth fog texture; sparse particle images resolve
    /// well with `[64, 32]`.
    public var windows: [Int] = [96, 64]
    /// Fractional overlap between neighbouring windows, in `[0, 1)`.
    public var overlap: Float = 0.5
    /// How many times the final window size is run (≥ 1).
    public var finalIterations = 2
    public var mask = MaskConfig()
    public var validation = ValidationConfig()

    public init() {}

    public func validate() throws(PIVError) {
        guard !windows.isEmpty else { throw .config("at least one window size is required") }
        for w in windows { try WindowSize.check(w) }
        guard overlap >= 0, overlap < 1 else { throw .config("overlap \(overlap) must be in [0, 1)") }
        guard finalIterations >= 1 else { throw .config("final iterations must be at least 1") }
    }

    /// The window size of every pass, in order.
    public var passes: [Int] {
        windows + Array(repeating: windows.last!, count: finalIterations - 1)
    }
}

/// Result of analysing one pair: the final-pass field plus its validation summary.
public struct PairResult: Sendable {
    public var field: VectorField
    public var summary: ValidationSummary
}

/// What one pass did (for parity checks against the Rust reference).
public struct PassTrace: Sendable {
    /// Window origins `[ax, ay, bx, by]` per grid cell, row-major.
    public var origins: [SIMD4<Float>]
    /// The correlation result before validation.
    public var raw: VectorField
    /// After validation (and replacement).
    public var validated: VectorField
}

/// Runs the configured multi-pass analysis on prepared frame pairs. Port of
/// `piv_core::PivAnalyzer`: window placement, predictor sampling and validation run on the
/// CPU exactly as in Rust; correlation and peak finding run on the GPU.
///
/// Holds GPU scratch buffers, so use one analyzer from one task at a time.
public final class PIVAnalyzer: @unchecked Sendable {
    public let config: PIVConfig
    public let path: CorrelationPath
    private let correlator: Correlator

    public init(config: PIVConfig, context: MetalContext, path: CorrelationPath = .auto) throws {
        try config.validate()
        self.config = config
        self.path = path
        correlator = Correlator(context: context)
        for n in Set(config.windows) { _ = try correlator.usesFused(n, path) }
    }

    /// Grid of the final pass for an image of the given size.
    public func outputGrid(height: Int, width: Int) throws -> Grid {
        try Grid(height: height, width: width, window: config.windows.last!, overlap: config.overlap)
    }

    /// Displacement field from frame `a` to frame `b`, in pixels.
    public func analyze(_ a: GPUFrame, _ b: GPUFrame) async throws -> PairResult {
        try await run(a, b, trace: false).0
    }

    /// Like ``analyze(_:_:)``, but also returns what each pass did.
    public func analyzeTraced(_ a: GPUFrame, _ b: GPUFrame) async throws -> (PairResult, [PassTrace]) {
        try await run(a, b, trace: true)
    }

    private func run(_ a: GPUFrame, _ b: GPUFrame, trace: Bool) async throws -> (PairResult, [PassTrace]) {
        guard a.width == b.width, a.height == b.height else {
            throw PIVError.shapeMismatch(a: .init(height: a.height, width: a.width), b: .init(height: b.height, width: b.width))
        }
        var result: PairResult?
        var traces: [PassTrace] = []
        let limits = ValidationLimits(maxDisplacement: Float(config.windows[0]) / 2)
        for n in config.passes {
            let grid = try Grid(height: a.height, width: a.width, window: n, overlap: config.overlap)
            let origins = Self.placeWindows(grid: grid, width: a.width, height: a.height, predictor: result?.field)
            let peaks = try await correlator.correlate(a: a, b: b, n: n, origins: origins, mask: config.mask, path: path)
            var field = Self.assemble(grid: grid, origins: origins, peaks: peaks)
            let raw = trace ? field : nil
            let summary = PIVKit.validate(&field, config: config.validation, limits: limits)
            if let raw { traces.append(PassTrace(origins: origins, raw: raw, validated: field)) }
            result = PairResult(field: field, summary: summary)
        }
        return (result!, traces)
    }

    /// Window origins for one pass. Each pass shifts the two windows symmetrically by the
    /// displacement predicted from the previous pass, so the vector stays located at the window
    /// centre. Near the border both windows move inward together rather than shrinking the
    /// offset. Fractional origins need one extra pixel for interpolation; a window as wide as
    /// the image has no room to move.
    static func placeWindows(grid: Grid, width: Int, height: Int, predictor: VectorField?) -> [SIMD4<Float>] {
        let n = grid.window
        let maxX = Float(max(width - n - 1, 0))
        let maxY = Float(max(height - n - 1, 0))
        func place(_ origin: Int, _ shift: Float, _ maxv: Float) -> (Float, Float) {
            let s = min(max(shift, -maxv), maxv)
            let a0 = min(max(Float(origin) - s / 2, max(-s, 0)), min(maxv - s, maxv))
            return (a0, a0 + s)
        }
        var out: [SIMD4<Float>] = []
        out.reserveCapacity(grid.count)
        for iy in 0..<grid.ny {
            for ix in 0..<grid.nx {
                let (x, y) = grid.windowOrigin(ix, iy)
                let p = predictor?.sample(x: grid.centerX(ix), y: grid.centerY(iy)) ?? (dx: 0, dy: 0)
                let (ax, bx) = place(x, p.dx, maxX)
                let (ay, by) = place(y, p.dy, maxY)
                out.append(SIMD4(ax, ay, bx, by))
            }
        }
        return out
    }

    static func assemble(grid: Grid, origins: [SIMD4<Float>], peaks: [WindowPeak]) -> VectorField {
        var f = VectorField(grid: grid)
        for k in 0..<grid.count where peaks[k].status == WindowPeak.ok {
            let o = origins[k], p = peaks[k]
            f.dx[k] = o.z - o.x + p.dx
            f.dy[k] = o.w - o.y + p.dy
            f.peakRatio[k] = p.ppr
            f.flags[k] = []
        }
        return f
    }
}
