import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Signal-mask thresholds on the (background-subtracted) raw intensity of a window, below which
/// it is treated as containing no seeding and is not correlated. Port of `MaskConfig`.
public struct MaskConfig: Hashable, Sendable, Codable {
    /// Minimum window mean intensity (0–1 scale).
    public var minMean: Float = 0
    /// Minimum window intensity standard deviation (0–1 scale).
    public var minStd: Float = 0
    public init(minMean: Float = 0, minStd: Float = 0) {
        self.minMean = minMean
        self.minStd = minStd
    }
}

/// Which correlation kernels to use.
public enum CorrelationPath: String, Hashable, Sendable, CaseIterable {
    /// The fastest measured option for each window size: fused up to 32 px, MPSGraph FFTs from
    /// 64 px, the custom batched FFT in between (e.g. 48 px).
    case auto
    /// One threadgroup per window, all in threadgroup memory (power-of-two windows ≤ 64 px).
    case fused
    /// Device-memory planes with the custom mixed-radix FFT (any 2^a·3^b window up to 128 px).
    case batched
    /// Device-memory planes with MPSGraph's FFT, around the same load / spectrum / peak kernels.
    case mpsGraph
}

/// Result of correlating one window pair (layout matches the Metal `WindowPeak`).
struct WindowPeak {
    var dx: Float
    var dy: Float
    var ppr: Float
    var height: Float
    var status: UInt32

    static let ok: UInt32 = 0
}

/// Layout matches the Metal `CorrParams`.
struct CorrelationParams {
    var n: UInt32
    var width: UInt32
    var height: UInt32
    var count: UInt32
    var minMean: Float
    var minStd: Float
    var log2n: UInt32
}

/// Layout matches the Metal `FFTLineParams`.
struct FFTLineParams {
    var n: UInt32
    var lines: UInt32
    var linesPerWindow: UInt32
    var windowStride: UInt32
    var lineStride: UInt32
    var elemStride: UInt32
    var linesPerTG: UInt32
    var threadsPerLine: UInt32
    var inverse: UInt32
    var stages: UInt32
    var radix: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    var checkStatus: UInt32
}

enum WindowSize {
    /// Radices (4s, then a 2, then 3s) for an n = 2^a·3^b; `nil` if n has another factor.
    static func radices(_ n: Int) -> [Int]? {
        var m = n, r: [Int] = []
        while m % 4 == 0 { r.append(4); m /= 4 }
        if m % 2 == 0 { r.append(2); m /= 2 }
        while m % 3 == 0 { r.append(3); m /= 3 }
        return m == 1 && r.count <= 8 ? r : nil
    }

    static func isPowerOfTwo(_ n: Int) -> Bool { n > 0 && n & (n - 1) == 0 }

    /// Largest window the batched FFT handles (one line per threadgroup buffer pair).
    static let maxBatched = 128
    /// Largest window whose packed complex plane fits in 32 KB of threadgroup memory.
    static let maxFused = 64

    static func check(_ n: Int) throws(PIVError) {
        guard n >= 8, n % 2 == 0 else { throw .config("window size \(n) must be even and ≥ 8") }
        guard n <= maxBatched, radices(n) != nil else {
            throw .config("window size \(n) is not supported on the GPU: it must be 2^a·3^b (e.g. 16, 24, 32, 48, 64, 96, 128) and at most \(maxBatched)")
        }
    }
}

/// Encodes window-pair correlations on the GPU. Owns reusable scratch buffers and MPSGraph
/// graphs, so one correlator must not be used from two tasks at once.
final class Correlator {
    let ctx: MetalContext
    private var twiddles: [Int: MTLBuffer] = [:]
    private var planes: MTLBuffer?
    private var planes2: MTLBuffer?
    private var graphs: [GraphKey: GraphFFT] = [:]

    init(context: MetalContext) { ctx = context }

    /// Largest window `auto` sends to the fused kernel. At 64 px the fused kernel's 32 KB
    /// threadgroups limit occupancy and it is slower than the batched path (M5 Max: 3.4 vs
    /// 2.7 ms for 1 888 windows); at 32 px it is faster (1.6 vs 2.9 ms).
    static let autoFusedMax = 32
    /// Smallest window `auto` sends to MPSGraph's FFT, which measured faster than the custom
    /// FFT at 64 and 96 px but slower at 48 px.
    static let autoMPSGraphMin = 64

    enum Kernel { case fused, custom, mpsGraph }

    /// The kernels `path` uses for windows of `n` px.
    func kernel(_ n: Int, _ path: CorrelationPath) throws -> Kernel {
        let fusable = WindowSize.isPowerOfTwo(n) && n <= WindowSize.maxFused
        switch path {
        case .auto:
            if fusable && n <= Self.autoFusedMax && (try? fusedThreads(n)) != nil { return .fused }
            return n >= Self.autoMPSGraphMin ? .mpsGraph : .custom
        case .batched: return .custom
        case .mpsGraph: return .mpsGraph
        case .fused:
            guard fusable else { throw PIVError.config("the fused path needs a power-of-two window ≤ \(WindowSize.maxFused) px, not \(n)") }
            return .fused
        }
    }

    /// Threads per fused threadgroup: n²/4 (each thread owns four plane cells), or as few as
    /// n²/8 if the pipeline can't run that many.
    private func fusedThreads(_ n: Int) throws -> Int {
        let limit = try ctx.pipeline("correlate_fused").maxTotalThreadsPerThreadgroup
        let want = n * n / 4
        if want <= limit { return want }
        guard n * n / 8 <= limit else { throw PIVError.metal("fused kernel limited to \(limit) threads; window \(n) needs \(n * n / 8)") }
        return n * n / 8
    }

    private func twiddleBuffer(_ n: Int) throws -> MTLBuffer {
        if let b = twiddles[n] { return b }
        let t: [SIMD2<Float>] = (0..<n).map { k in
            let a = -2 * Double.pi * Double(k) / Double(n)
            return SIMD2(Float(cos(a)), Float(sin(a)))
        }
        let b = try ctx.makeBuffer(t)
        twiddles[n] = b
        return b
    }

    private func scratch(_ buffer: inout MTLBuffer?, bytes: Int) throws -> MTLBuffer {
        if let b = buffer, b.length >= bytes { return b }
        let b = try ctx.makeBuffer(length: bytes)
        buffer = b
        return b
    }

    /// Encodes correlation of `origins.count` window pairs of size `n` into `cb` and returns
    /// the buffer of `WindowPeak`s it will fill.
    func encode(
        _ cb: MPSCommandBuffer, a: GPUFrame, b: GPUFrame, n: Int, origins: [SIMD4<Float>],
        mask: MaskConfig, path: CorrelationPath
    ) throws -> MTLBuffer {
        try WindowSize.check(n)
        let count = origins.count
        let originBuf = try ctx.makeBuffer(origins)
        let out = try ctx.makeBuffer(length: count * MemoryLayout<WindowPeak>.stride)
        guard count > 0 else { return out }
        var params = CorrelationParams(
            n: UInt32(n), width: UInt32(a.width), height: UInt32(a.height), count: UInt32(count),
            minMean: mask.minMean, minStd: mask.minStd, log2n: UInt32(n.trailingZeroBitCount)
        )
        func compute(_ body: (MTLComputeCommandEncoder) throws -> Void) throws {
            guard let enc = cb.makeComputeCommandEncoder() else { throw PIVError.metal("cannot create an encoder") }
            defer { enc.endEncoding() }
            try body(enc)
        }
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let kind = try kernel(n, path)

        if kind == .fused {
            let threads = try fusedThreads(n)
            try compute { enc in
                enc.setComputePipelineState(try ctx.pipeline("correlate_fused"))
                enc.setBuffers([a.image, b.image, a.stats, b.stats, originBuf], offsets: [0, 0, 0, 0, 0], range: 0..<5)
                enc.setBytes(&params, length: MemoryLayout<CorrelationParams>.stride, index: 5)
                enc.setBuffer(try twiddleBuffer(n), offset: 0, index: 6)
                enc.setBuffer(out, offset: 0, index: 7)
                enc.setThreadgroupMemoryLength(n * n * 8, index: 0)
                enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            }
            return out
        }

        // Batched: windows → p0; forward FFT → p1 (in place for the custom FFT); packed cross
        // spectrum → p0; inverse FFT of the ⌈count/2⌉ packed planes → p1; peaks from p1.
        let pairs = (count + 1) / 2
        let planeBytes = count * n * n * MemoryLayout<SIMD2<Float>>.stride
        let p0 = try scratch(&planes, bytes: planeBytes)
        let p1 = try scratch(&planes2, bytes: planeBytes)

        try compute { enc in
            enc.setComputePipelineState(try ctx.pipeline("load_windows"))
            enc.setBuffers([a.image, b.image, a.stats, b.stats, originBuf], offsets: [0, 0, 0, 0, 0], range: 0..<5)
            enc.setBytes(&params, length: MemoryLayout<CorrelationParams>.stride, index: 5)
            enc.setBuffer(p0, offset: 0, index: 6)
            enc.setBuffer(out, offset: 0, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: tg)
            if kind == .custom {
                try encodeFFT(enc, planes: p0, status: out, n: n, count: count, inverse: false)
            }
        }
        let spectra: MTLBuffer
        if kind == .mpsGraph {
            try graphFFT(n: n, count: count, inverse: false).encode(cb, from: p0, to: p1)
            spectra = p1
        } else {
            spectra = p0
        }
        let packed = spectra === p0 ? p1 : p0
        try compute { enc in
            enc.setComputePipelineState(try ctx.pipeline("cross_spectrum_packed"))
            enc.setBuffer(spectra, offset: 0, index: 0)
            enc.setBytes(&params, length: MemoryLayout<CorrelationParams>.stride, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            enc.setBuffer(packed, offset: 0, index: 3)
            enc.dispatchThreads(MTLSize(width: pairs * n * n, height: 1, depth: 1), threadsPerThreadgroup: tg)
            if kind == .custom {
                try encodeFFT(enc, planes: packed, status: nil, n: n, count: pairs, inverse: true)
            }
        }
        let correlations: MTLBuffer
        if kind == .mpsGraph {
            correlations = packed === p0 ? p1 : p0
            try graphFFT(n: n, count: pairs, inverse: true).encode(cb, from: packed, to: correlations)
        } else {
            correlations = packed
        }
        try compute { enc in
            enc.setComputePipelineState(try ctx.pipeline("find_peaks"))
            enc.setBuffer(correlations, offset: 0, index: 0)
            enc.setBytes(&params, length: MemoryLayout<CorrelationParams>.stride, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: tg)
        }
        return out
    }

    /// Unnormalized 2-D DFT of `count` n × n planes in place with the custom FFT: all rows,
    /// then all columns. With a `status` buffer, planes of windows that weren't correlated are
    /// skipped.
    func encodeFFT(
        _ enc: MTLComputeCommandEncoder, planes: MTLBuffer, status: MTLBuffer?, n: Int, count: Int, inverse: Bool
    ) throws {
        let radices = WindowSize.radices(n)!
        var r = [UInt32](repeating: 0, count: 8)
        for (i, x) in radices.enumerated() { r[i] = UInt32(x) }
        let tpl = n / 2
        let linesPerTG = max(1, 256 / tpl)
        let lines = count * n
        for rows in [true, false] {
            var fp = FFTLineParams(
                n: UInt32(n), lines: UInt32(lines), linesPerWindow: UInt32(n), windowStride: UInt32(n * n),
                lineStride: UInt32(rows ? n : 1), elemStride: UInt32(rows ? 1 : n),
                linesPerTG: UInt32(linesPerTG), threadsPerLine: UInt32(tpl), inverse: inverse ? 1 : 0,
                stages: UInt32(radices.count), radix: (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]),
                checkStatus: status == nil ? 0 : 1
            )
            enc.setComputePipelineState(try ctx.pipeline("fft_lines"))
            enc.setBuffer(planes, offset: 0, index: 0)
            enc.setBytes(&fp, length: MemoryLayout<FFTLineParams>.stride, index: 1)
            enc.setBuffer(try twiddleBuffer(n), offset: 0, index: 2)
            enc.setBuffer(status ?? planes, offset: 0, index: 3)
            enc.setThreadgroupMemoryLength(linesPerTG * 2 * n * 8, index: 0)
            enc.dispatchThreadgroups(
                MTLSize(width: (lines + linesPerTG - 1) / linesPerTG, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: linesPerTG * tpl, height: 1, depth: 1))
        }
    }

    private struct GraphKey: Hashable { let n: Int, count: Int, inverse: Bool }

    private func graphFFT(n: Int, count: Int, inverse: Bool) throws -> GraphFFT {
        let key = GraphKey(n: n, count: count, inverse: inverse)
        if let g = graphs[key] { return g }
        let g = GraphFFT(n: n, count: count, inverse: inverse)
        graphs[key] = g
        return g
    }

    /// Correlates and waits for the result.
    func correlate(
        a: GPUFrame, b: GPUFrame, n: Int, origins: [SIMD4<Float>], mask: MaskConfig, path: CorrelationPath
    ) async throws -> [WindowPeak] {
        let cb = MPSCommandBuffer(from: ctx.queue)
        let out = try encode(cb, a: a, b: b, n: n, origins: origins, mask: mask, path: path)
        try await ctx.run(cb)
        let p = out.contents().bindMemory(to: WindowPeak.self, capacity: origins.count)
        return Array(UnsafeBufferPointer(start: p, count: origins.count))
    }
}

/// An unnormalized batched 2-D FFT over `[count, n, n]` complex planes with MPSGraph.
final class GraphFFT {
    let graph = MPSGraph()
    let input: MPSGraphTensor
    let output: MPSGraphTensor
    let shape: [NSNumber]

    init(n: Int, count: Int, inverse: Bool) {
        shape = [count, n, n].map { NSNumber(value: $0) }
        input = graph.placeholder(shape: shape, dataType: .complexFloat32, name: nil)
        let d = MPSGraphFFTDescriptor()
        d.inverse = inverse
        d.scalingMode = .none
        output = graph.fastFourierTransform(input, axes: [1, 2], descriptor: d, name: nil)
    }

    func encode(_ cb: MPSCommandBuffer, from: MTLBuffer, to: MTLBuffer) throws {
        graph.encode(
            to: cb,
            feeds: [input: MPSGraphTensorData(from, shape: shape, dataType: .complexFloat32)],
            targetOperations: nil,
            resultsDictionary: [output: MPSGraphTensorData(to, shape: shape, dataType: .complexFloat32)],
            executionDescriptor: nil
        )
    }
}
