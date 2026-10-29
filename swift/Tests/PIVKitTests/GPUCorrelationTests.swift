import Foundation
import Metal
import Testing
@testable import PIVKit

let gpu: MetalContext? = try? MetalContext()

/// Deterministic uniform numbers for test images.
struct TestRNG {
    var state: UInt64
    mutating func uniform() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(Double(state >> 11) / Double(UInt64(1) << 53))
    }
}

@Suite(.enabled(if: gpu != nil, "needs a Metal device"))
struct GPUCorrelationTests {
    let ctx = gpu!

    @Test func kernelsCompile() throws {
        for name in ["correlate_fused", "load_windows", "fft_lines", "cross_spectrum", "find_peaks"] {
            _ = try ctx.pipeline(name)
        }
    }

    /// A periodic random pattern shifted circularly correlates perfectly (Rust:
    /// `integer_shift_is_exact_for_circular_content`).
    @Test(arguments: [(32, CorrelationPath.fused), (32, .batched), (64, .fused), (64, .batched),
                      (24, .batched), (48, .batched), (96, .batched), (128, .batched)])
    func integerShiftIsExactForCircularContent(_ n: Int, _ path: CorrelationPath) async throws {
        var rng = TestRNG(state: 7)
        let a = (0..<n * n).map { _ in rng.uniform() }
        let (sx, sy) = (5, n - 3) // +5 in x, −3 in y
        var b = [Float](repeating: 0, count: n * n)
        for y in 0..<n { for x in 0..<n { b[y * n + x] = a[((y + n - sy) % n) * n + (x + n - sx) % n] } }
        let fa = try ctx.makeFrame(image: a, stats: a, width: n, height: n)
        let fb = try ctx.makeFrame(image: b, stats: b, width: n, height: n)
        let p = try await Correlator(context: ctx).correlate(
            a: fa, b: fb, n: n, origins: [SIMD4(0, 0, 0, 0)], mask: MaskConfig(minMean: -1, minStd: -1), path: path)[0]
        #expect(p.status == 0)
        #expect(abs(p.dx - 5) < 0.02 && abs(p.dy + 3) < 0.02, "\(p)")
        #expect(abs(p.height - 1) < 1e-4, "\(p)")
        #expect(p.ppr > 2)
    }

    @Test(arguments: [CorrelationPath.fused, .batched])
    func flatWindowHasNoPeak(_ path: CorrelationPath) async throws {
        let img = [Float](repeating: 0.3, count: 32 * 32)
        let f = try ctx.makeFrame(image: img, stats: img, width: 32, height: 32)
        let p = try await Correlator(context: ctx).correlate(
            a: f, b: f, n: 32, origins: [.zero], mask: MaskConfig(), path: path)[0]
        #expect(p.status == 2)
    }

    @Test(arguments: [CorrelationPath.fused, .batched])
    func maskRejectsDarkWindows(_ path: CorrelationPath) async throws {
        var rng = TestRNG(state: 3)
        let img = (0..<32 * 32).map { _ in 0.005 * rng.uniform() }
        let f = try ctx.makeFrame(image: img, stats: img, width: 32, height: 32)
        let p = try await Correlator(context: ctx).correlate(
            a: f, b: f, n: 32, origins: [.zero], mask: MaskConfig(minMean: 0.01, minStd: 0.01), path: path)[0]
        #expect(p.status == 1)
    }

    /// Rust `peak_ratio_uses_local_maxima`: neighbours of a broad maximum are not a second peak.
    @Test func peakRatioUsesLocalMaxima() async throws {
        let n = 16
        var c = [Float](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                let (dy, dx) = (Float(min(y, n - y)), Float(min(x, n - x)))
                c[y * n + x] = exp(-(dx * dx + dy * dy) / 8)
            }
        }
        c[8 * n + 8] = 0.25
        c[2 * n] += 0.05
        c[3 * n] -= 0.05
        let p = try await findPeaks(planes: [c], n: n)[0]
        #expect((p.dx, p.dy) == (0, 0))
        #expect(abs(p.ppr - 4) < 1e-3, "\(p)")
    }

    /// Runs the `find_peaks` kernel on given real planes.
    func findPeaks(planes: [[Float]], n: Int) async throws -> [WindowPeak] {
        let data = planes.flatMap { $0.map { SIMD2<Float>($0, 0) } }
        let planeBuf = try ctx.makeBuffer(data)
        let out = try ctx.makeBuffer(planes.map { _ in WindowPeak(dx: 0, dy: 0, ppr: 0, height: 1, status: 0) })
        var params = CorrelationParams(n: UInt32(n), width: 0, height: 0, count: UInt32(planes.count), minMean: 0, minStd: 0, log2n: 0)
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(try ctx.pipeline("find_peaks"))
        enc.setBuffer(planeBuf, offset: 0, index: 0)
        enc.setBytes(&params, length: MemoryLayout<CorrelationParams>.stride, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: planes.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        try await ctx.run(cb)
        let p = out.contents().bindMemory(to: WindowPeak.self, capacity: planes.count)
        return Array(UnsafeBufferPointer(start: p, count: planes.count))
    }

    @Test func layoutsMatchMetal() {
        #expect(MemoryLayout<WindowPeak>.stride == 20)
        #expect(MemoryLayout<CorrelationParams>.stride == 28)
        #expect(MemoryLayout<FFTLineParams>.stride == 72)
    }

    @Test func windowSizes() {
        #expect(WindowSize.radices(96) == [4, 4, 2, 3])
        #expect(WindowSize.radices(64) == [4, 4, 4])
        #expect(WindowSize.radices(80) == nil)
        #expect(throws: PIVError.self) { try WindowSize.check(80) }
        #expect(throws: PIVError.self) { try WindowSize.check(192) }
    }
}
