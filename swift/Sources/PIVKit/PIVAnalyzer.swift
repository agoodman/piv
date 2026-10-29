import Foundation
import Metal
import MetalPerformanceShadersGraph

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
/// `piv_core::PivAnalyzer`. Every stage runs on the GPU: window placement from the previous
/// pass's field, correlation and peak finding, and validation, all passes of a pair in one
/// command buffer, so the CPU only reads back the final field. The placement and validation
/// kernels repeat Rust's f32 operations in the same order.
///
/// Holds GPU scratch buffers, so use one analyzer from one task at a time.
public final class PIVAnalyzer: @unchecked Sendable {
    public let config: PIVConfig
    public let path: CorrelationPath
    private let correlator: Correlator
    private let ctx: MetalContext

    public init(config: PIVConfig, context: MetalContext, path: CorrelationPath = .auto) throws {
        try config.validate()
        self.config = config
        self.path = path
        ctx = context
        correlator = Correlator(context: context)
        for n in Set(config.windows) { _ = try correlator.kernel(n, path) }
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

    /// GPU buffers for one pass's field.
    struct PassBuffers {
        let grid: Grid
        let origins: MTLBuffer
        let dx, dy, ppr: MTLBuffer       // raw (before replacement)
        let flags1, flags2: MTLBuffer    // after the per-vector tests; after the median test
        let dxOut, dyOut, flagsOut: MTLBuffer // validated

        init(_ ctx: MetalContext, grid: Grid) throws {
            self.grid = grid
            let n = grid.count, f = n * MemoryLayout<Float>.stride
            origins = try ctx.makeBuffer(length: n * MemoryLayout<SIMD4<Float>>.stride)
            (dx, dy, ppr) = (try ctx.makeBuffer(length: f), try ctx.makeBuffer(length: f), try ctx.makeBuffer(length: f))
            (flags1, flags2, flagsOut) = (try ctx.makeBuffer(length: n), try ctx.makeBuffer(length: n), try ctx.makeBuffer(length: n))
            (dxOut, dyOut) = (try ctx.makeBuffer(length: f), try ctx.makeBuffer(length: f))
        }

        func field(validated: Bool) -> VectorField {
            let n = grid.count
            let fl = (validated ? flagsOut : flags1).contents().bindMemory(to: UInt8.self, capacity: n)
            var flags = (0..<n).map { VectorFlags(rawValue: fl[$0]) }
            if !validated { flags = flags.map { $0.intersection(.noSignal) } } // before validation
            return VectorField(
                grid: grid, dx: (validated ? dxOut : dx).floats(n), dy: (validated ? dyOut : dy).floats(n),
                peakRatio: ppr.floats(n), flags: flags)
        }

        var originArray: [SIMD4<Float>] {
            Array(UnsafeBufferPointer(start: origins.contents().bindMemory(to: SIMD4<Float>.self, capacity: grid.count), count: grid.count))
        }
    }

    /// Per-pass buffers, reused from pair to pair (the analyzer runs one pair at a time).
    private var buffers: [Int: PassBuffers] = [:]
    private var bufferSize: (Int, Int)?

    private func passBuffers(_ i: Int, grid: Grid, size: (Int, Int)) throws -> PassBuffers {
        if let s = bufferSize, s != size { buffers = [:] }
        bufferSize = size
        if let b = buffers[i], b.grid == grid { return b }
        let b = try PassBuffers(ctx, grid: grid)
        buffers[i] = b
        return b
    }

    func fieldParams(grid: Grid, predictor: Grid?, width: Int, height: Int) -> FieldParams {
        let v = config.validation
        let pg = predictor ?? grid
        return FieldParams(
            nx: UInt32(grid.nx), ny: UInt32(grid.ny), step: UInt32(grid.step), window: UInt32(grid.window),
            x0: UInt32(grid.x0), y0: UInt32(grid.y0),
            pnx: UInt32(pg.nx), pny: UInt32(pg.ny), pstep: UInt32(pg.step), pwindow: UInt32(pg.window),
            px0: UInt32(pg.x0), py0: UInt32(pg.y0), hasPredictor: predictor == nil ? 0 : 1,
            maxX: Float(max(width - grid.window - 1, 0)), maxY: Float(max(height - grid.window - 1, 0)),
            maxDisplacement: Float(config.windows[0]) / 2,
            pprThreshold: v.pprThreshold, medianThreshold: v.medianThreshold, medianEps: v.medianEps,
            replace: v.replace ? 1 : 0)
    }

    /// Encodes placement (from `predictor`, the previous pass's buffers) into `enc`.
    func encodePlace(_ enc: MTLComputeCommandEncoder, _ pb: PassBuffers, predictor: PassBuffers?, params: inout FieldParams) throws {
        enc.setComputePipelineState(try ctx.pipeline("place_windows"))
        enc.setBuffer(predictor?.dxOut ?? pb.dxOut, offset: 0, index: 0)
        enc.setBuffer(predictor?.dyOut ?? pb.dyOut, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<FieldParams>.stride, index: 2)
        enc.setBuffer(pb.origins, offset: 0, index: 3)
        dispatch(enc, pb.grid.count)
    }

    /// Encodes assembly from `peaks` and the two validation kernels into `enc`.
    func encodeValidate(_ enc: MTLComputeCommandEncoder, _ pb: PassBuffers, peaks: MTLBuffer?, params: inout FieldParams) throws {
        let n = pb.grid.count
        if let peaks {
            enc.setComputePipelineState(try ctx.pipeline("assemble_field"))
            enc.setBuffer(pb.origins, offset: 0, index: 0)
            enc.setBuffer(peaks, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<FieldParams>.stride, index: 2)
            enc.setBuffers([pb.dx, pb.dy, pb.ppr, pb.flags1], offsets: [0, 0, 0, 0], range: 3..<7)
            dispatch(enc, n)
        }
        enc.setComputePipelineState(try ctx.pipeline("median_test"))
        enc.setBuffers([pb.dx, pb.dy, pb.flags1, pb.flags2], offsets: [0, 0, 0, 0], range: 0..<4)
        enc.setBytes(&params, length: MemoryLayout<FieldParams>.stride, index: 4)
        dispatch(enc, n)
        enc.setComputePipelineState(try ctx.pipeline("replace_outliers"))
        enc.setBuffers([pb.dx, pb.dy, pb.flags2, pb.dxOut, pb.dyOut, pb.flagsOut], offsets: [0, 0, 0, 0, 0, 0], range: 0..<6)
        enc.setBytes(&params, length: MemoryLayout<FieldParams>.stride, index: 6)
        dispatch(enc, n)
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ n: Int) {
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    private func run(_ a: GPUFrame, _ b: GPUFrame, trace: Bool) async throws -> (PairResult, [PassTrace]) {
        guard a.width == b.width, a.height == b.height else {
            throw PIVError.shapeMismatch(a: .init(height: a.height, width: a.width), b: .init(height: b.height, width: b.width))
        }
        let cb = MPSCommandBuffer(from: ctx.queue)
        var previous: PassBuffers?
        var all: [PassBuffers] = []
        for (i, n) in config.passes.enumerated() {
            let grid = try Grid(height: a.height, width: a.width, window: n, overlap: config.overlap)
            let pb = try passBuffers(i, grid: grid, size: (a.width, a.height))
            var params = fieldParams(grid: grid, predictor: previous?.grid, width: a.width, height: a.height)
            guard let enc = cb.makeComputeCommandEncoder() else { throw PIVError.metal("no encoder") }
            try encodePlace(enc, pb, predictor: previous, params: &params)
            enc.endEncoding()
            let peaks = try correlator.encode(cb, a: a, b: b, n: n, origins: pb.origins, count: grid.count, mask: config.mask, path: path)
            guard let enc2 = cb.makeComputeCommandEncoder() else { throw PIVError.metal("no encoder") }
            try encodeValidate(enc2, pb, peaks: peaks, params: &params)
            enc2.endEncoding()
            previous = pb
            all.append(pb)
        }
        try await ctx.run(cb)
        let field = all.last!.field(validated: true)
        let traces = trace ? all.map { PassTrace(origins: $0.originArray, raw: $0.field(validated: false), validated: $0.field(validated: true)) } : []
        return (PairResult(field: field, summary: Self.summary(field.flags)), traces)
    }

    /// Validation counts for a final field (each vector counted once per flag, as `validate`
    /// counts them).
    static func summary(_ flags: [VectorFlags]) -> ValidationSummary {
        var s = ValidationSummary()
        for f in flags {
            if f.contains(.noSignal) { s.noSignal += 1; continue }
            if f.contains(.lowPPR) { s.lowPPR += 1 }
            if f.contains(.medianOutlier) { s.medianOutliers += 1 }
            if f.contains(.outOfRange) { s.outOfRange += 1 }
            if f.contains(.replaced) { s.replaced += 1 }
        }
        return s
    }

    /// GPU validation of a given raw field (for parity tests): the per-vector tests, the median
    /// test and replacement, exactly as in a pass.
    func validateOnGPU(_ raw: VectorField) async throws -> VectorField {
        let pb = try PassBuffers(ctx, grid: raw.grid)
        // Per-vector flags as assemble_field would set them.
        let maxD = Float(config.windows[0]) / 2
        let f1: [UInt8] = raw.flags.indices.map { k in
            if raw.flags[k].contains(.noSignal) { return VectorFlags.noSignal.rawValue }
            var f = VectorFlags()
            if hypotf(raw.dx[k], raw.dy[k]) > maxD { f.insert(.outOfRange) }
            if raw.peakRatio[k] < config.validation.pprThreshold { f.insert(.lowPPR) }
            return f.rawValue
        }
        for (buf, values) in [(pb.dx, raw.dx), (pb.dy, raw.dy), (pb.ppr, raw.peakRatio)] {
            values.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        f1.withUnsafeBytes { pb.flags1.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        var params = fieldParams(grid: raw.grid, predictor: nil, width: 0, height: 0)
        guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw PIVError.metal("no encoder") }
        try encodeValidate(enc, pb, peaks: nil, params: &params)
        enc.endEncoding()
        try await ctx.run(cb)
        return pb.field(validated: true)
    }

    /// GPU window placement from a given predictor field (for parity tests).
    func placeOnGPU(grid: Grid, width: Int, height: Int, predictor: VectorField?) async throws -> [SIMD4<Float>] {
        let pb = try PassBuffers(ctx, grid: grid)
        var pred: PassBuffers?
        if let p = predictor {
            let q = try PassBuffers(ctx, grid: p.grid)
            p.dx.withUnsafeBytes { q.dxOut.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            p.dy.withUnsafeBytes { q.dyOut.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            pred = q
        }
        var params = fieldParams(grid: grid, predictor: predictor?.grid, width: width, height: height)
        guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw PIVError.metal("no encoder") }
        try encodePlace(enc, pb, predictor: pred, params: &params)
        enc.endEncoding()
        try await ctx.run(cb)
        return pb.originArray
    }

    /// Window origins for one pass, on the CPU (the reference for the GPU placement kernel).
    /// Each pass shifts the two windows symmetrically by the
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

/// Layout matches the Metal `FieldParams`.
struct FieldParams {
    var nx, ny, step, window, x0, y0: UInt32
    var pnx, pny, pstep, pwindow, px0, py0: UInt32
    var hasPredictor: UInt32
    var maxX, maxY: Float
    var maxDisplacement: Float
    var pprThreshold, medianThreshold, medianEps: Float
    var replace: UInt32
}
