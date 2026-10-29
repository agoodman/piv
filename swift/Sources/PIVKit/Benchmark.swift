import Foundation
import Metal
import MetalPerformanceShadersGraph

/// GPU timings for choosing correlation kernels (`pivkit bench`). Times are medians of GPU
/// execution time from command-buffer timestamps, so they exclude CPU encoding and waiting.
public enum CorrelationBenchmark {
    public struct Row: Sendable {
        public var label: String
        public var windows: Int
        public var milliseconds: Double
    }

    /// A 1920 × 1080 pair of smooth random texture, the second shifted by (3, 1) px.
    static func testFrames(_ ctx: MetalContext, width: Int = 1920, height: Int = 1080) throws -> (GPUFrame, GPUFrame) {
        var rng = SplitMix(seed: 42)
        var noise = (0..<width * height).map { _ in Float(rng.next() >> 40) / Float(1 << 24) }
        // Two box blurs: texture a few pixels across, like fog.
        for _ in 0..<2 {
            var out = noise
            for y in 1..<height - 1 {
                for x in 1..<width - 1 {
                    var s: Float = 0
                    for dy in -1...1 { for dx in -1...1 { s += noise[(y + dy) * width + x + dx] } }
                    out[y * width + x] = s / 9
                }
            }
            noise = out
        }
        var b = noise
        for y in 0..<height { for x in 0..<width { b[y * width + x] = noise[((y + height - 1) % height) * width + (x + width - 3) % width] } }
        return (try ctx.makeFrame(image: noise, stats: noise, width: width, height: height),
                try ctx.makeFrame(image: b, stats: b, width: width, height: height))
    }

    static func gpuMilliseconds(_ ctx: MetalContext, repeats: Int, _ encode: (MTLCommandBuffer) throws -> Void) async throws -> Double {
        var times: [Double] = []
        for i in 0..<(repeats + 2) {
            guard let cb = ctx.queue.makeCommandBuffer() else { throw PIVError.metal("no command buffer") }
            try encode(cb)
            try await ctx.run(cb)
            if i >= 2 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3) } // skip warm-up
        }
        times.sort()
        return times[times.count / 2]
    }

    /// Per-pass correlation at 1080p: fused vs batched kernels, the batched FFT alone, and
    /// MPSGraph's FFT on the same batch.
    public static func correlation(context ctx: MetalContext, repeats: Int = 9) async throws -> [Row] {
        let (a, b) = try testFrames(ctx)
        var rows: [Row] = []
        for n in [96, 64, 48, 32] {
            let grid = try Grid(height: a.height, width: a.width, window: n, overlap: 0.5)
            let origins = PIVAnalyzer.placeWindows(grid: grid, width: a.width, height: a.height, predictor: nil)
            let count = origins.count
            let corr = Correlator(context: ctx)
            let mask = MaskConfig()
            if try corr.usesFused(n, .auto) {
                let t = try await gpuMilliseconds(ctx, repeats: repeats) { cb in
                    _ = try corr.encode(cb, a: a, b: b, n: n, origins: origins, mask: mask, path: .fused)
                }
                rows.append(Row(label: "fused correlate (load+FFT+peak)", windows: n, milliseconds: t))
            }
            let t = try await gpuMilliseconds(ctx, repeats: repeats) { cb in
                _ = try corr.encode(cb, a: a, b: b, n: n, origins: origins, mask: mask, path: .batched)
            }
            rows.append(Row(label: "batched correlate (load+FFT+peak)", windows: n, milliseconds: t))

            for stage in BatchedStage.allCases {
                let ts = try await gpuMilliseconds(ctx, repeats: repeats) { cb in
                    _ = try corr.encode(cb, a: a, b: b, n: n, origins: origins, mask: mask, path: .batched, stages: [stage])
                }
                rows.append(Row(label: "  batched stage \(stage)", windows: n, milliseconds: ts))
            }

            // FFT stage alone: forward + inverse over every plane.
            let planes = try ctx.makeBuffer(length: count * n * n * 8)
            let status = try ctx.makeBuffer([WindowPeak](repeating: WindowPeak(dx: 0, dy: 0, ppr: 0, height: 1, status: 0), count: count))
            let tf = try await gpuMilliseconds(ctx, repeats: repeats) { cb in
                guard let enc = cb.makeComputeCommandEncoder() else { throw PIVError.metal("no encoder") }
                try corr.encodeFFT(enc, planes: planes, status: status, n: n, count: count, inverse: false)
                try corr.encodeFFT(enc, planes: planes, status: status, n: n, count: count, inverse: true)
                enc.endEncoding()
            }
            rows.append(Row(label: "batched FFT only (fwd+inv)", windows: n, milliseconds: tf))

            do {
                let (th, dmax) = try await hybrid(ctx, a: a, b: b, n: n, origins: origins, repeats: repeats)
                rows.append(Row(label: "hybrid correlate (MPSGraph FFT), max |Δd| vs batched \(String(format: "%.1e", dmax))", windows: n, milliseconds: th))
            } catch {
                rows.append(Row(label: "hybrid failed: \(error)", windows: n, milliseconds: .nan))
            }
            if let tg = try? await mpsGraphFFT(ctx, planes: planes, n: n, count: count, repeats: repeats) {
                rows.append(Row(label: "MPSGraph FFT only (fwd+inv)", windows: n, milliseconds: tg))
            } else {
                rows.append(Row(label: "MPSGraph FFT only (fwd+inv): failed", windows: n, milliseconds: .nan))
            }
        }
        return rows
    }

    static func mpsGraphFFT(_ ctx: MetalContext, planes: MTLBuffer, n: Int, count: Int, repeats: Int) async throws -> Double {
        let graph = MPSGraph()
        let shape: [NSNumber] = [count, n, n].map { NSNumber(value: $0) }
        let x = graph.placeholder(shape: shape, dataType: .complexFloat32, name: nil)
        let fwd = MPSGraphFFTDescriptor()
        fwd.inverse = false
        fwd.scalingMode = .none
        let inv = MPSGraphFFTDescriptor()
        inv.inverse = true
        inv.scalingMode = .none
        let f = graph.fastFourierTransform(x, axes: [1, 2], descriptor: fwd, name: nil)
        let g = graph.fastFourierTransform(f, axes: [1, 2], descriptor: inv, name: nil)
        let data = MPSGraphTensorData(planes, shape: shape, dataType: .complexFloat32)
        var times: [Double] = []
        for i in 0..<(repeats + 2) {
            let cb = MPSCommandBuffer(from: ctx.queue)
            _ = graph.encode(to: cb, feeds: [x: data], targetTensors: [g], targetOperations: nil, executionDescriptor: nil)
            let root = cb.rootCommandBuffer
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                root.addCompletedHandler { _ in c.resume() }
                cb.commit()
            }
            if let e = root.error { throw PIVError.metal("MPSGraph: \(e)") }
            if i >= 2 { times.append((root.gpuEndTime - root.gpuStartTime) * 1e3) }
        }
        times.sort()
        return times[times.count / 2]
    }

    /// My load / cross-spectrum / peak kernels around MPSGraph's forward and inverse FFTs, in
    /// one command buffer. Returns the GPU time and the largest displacement difference from
    /// the all-custom batched path.
    static func hybrid(_ ctx: MetalContext, a: GPUFrame, b: GPUFrame, n: Int, origins: [SIMD4<Float>], repeats: Int) async throws -> (Double, Float) {
        let count = origins.count
        let originBuf = try ctx.makeBuffer(origins)
        let out = try ctx.makeBuffer(length: count * MemoryLayout<WindowPeak>.stride)
        let p0 = try ctx.makeBuffer(length: count * n * n * 8)
        let p1 = try ctx.makeBuffer(length: count * n * n * 8)
        var params = CorrelationParams(n: UInt32(n), width: UInt32(a.width), height: UInt32(a.height), count: UInt32(count), minMean: 0, minStd: 0, log2n: 0)

        let graph = MPSGraph()
        let shape: [NSNumber] = [count, n, n].map { NSNumber(value: $0) }
        let xf = graph.placeholder(shape: shape, dataType: .complexFloat32, name: nil)
        let df = MPSGraphFFTDescriptor(); df.inverse = false; df.scalingMode = .none
        let yf = graph.fastFourierTransform(xf, axes: [1, 2], descriptor: df, name: nil)
        let di = MPSGraphFFTDescriptor(); di.inverse = true; di.scalingMode = .none
        let xi = graph.placeholder(shape: shape, dataType: .complexFloat32, name: nil)
        let yi = graph.fastFourierTransform(xi, axes: [1, 2], descriptor: di, name: nil)
        let t0 = MPSGraphTensorData(p0, shape: shape, dataType: .complexFloat32)
        let t1 = MPSGraphTensorData(p1, shape: shape, dataType: .complexFloat32)

        func compute(_ cb: MTLCommandBuffer, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
            let enc = cb.makeComputeCommandEncoder()!
            try body(enc)
            enc.endEncoding()
        }
        var times: [Double] = []
        for i in 0..<(repeats + 2) {
            let cb = MPSCommandBuffer(from: ctx.queue)
            try compute(cb) { enc in
                enc.setComputePipelineState(try ctx.pipeline("load_windows"))
                enc.setBuffers([a.image, b.image, a.stats, b.stats, originBuf], offsets: [0, 0, 0, 0, 0], range: 0..<5)
                enc.setBytes(&params, length: MemoryLayout<CorrelationParams>.stride, index: 5)
                enc.setBuffer(p0, offset: 0, index: 6)
                enc.setBuffer(out, offset: 0, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            graph.encode(to: cb, feeds: [xf: t0], targetOperations: nil, resultsDictionary: [yf: t1], executionDescriptor: nil)
            try compute(cb) { enc in
                enc.setComputePipelineState(try ctx.pipeline("cross_spectrum"))
                enc.setBuffer(p1, offset: 0, index: 0)
                enc.setBytes(&params, length: MemoryLayout<CorrelationParams>.stride, index: 1)
                enc.setBuffer(out, offset: 0, index: 2)
                enc.dispatchThreads(MTLSize(width: count * n * n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            graph.encode(to: cb, feeds: [xi: t1], targetOperations: nil, resultsDictionary: [yi: t0], executionDescriptor: nil)
            try compute(cb) { enc in
                enc.setComputePipelineState(try ctx.pipeline("find_peaks"))
                enc.setBuffer(p0, offset: 0, index: 0)
                enc.setBytes(&params, length: MemoryLayout<CorrelationParams>.stride, index: 1)
                enc.setBuffer(out, offset: 0, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            let root = cb.rootCommandBuffer
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                root.addCompletedHandler { _ in c.resume() }
                cb.commit()
            }
            if let e = root.error { throw PIVError.metal("hybrid: \(e)") }
            if i >= 2 { times.append((root.gpuEndTime - root.gpuStartTime) * 1e3) }
        }
        times.sort()
        let hy = Array(UnsafeBufferPointer(start: out.contents().bindMemory(to: WindowPeak.self, capacity: count), count: count))
        let ref = try await Correlator(context: ctx).correlate(a: a, b: b, n: n, origins: origins, mask: MaskConfig(), path: .batched)
        var dmax: Float = 0
        for k in 0..<count where ref[k].status == 0 {
            dmax = max(dmax, abs(hy[k].dx - ref[k].dx), abs(hy[k].dy - ref[k].dy))
        }
        return (times[times.count / 2], dmax)
    }

    /// One full pair with the default settings ([96, 64], two final iterations) at 1080p, per
    /// path: wall-clock time per pair including CPU window placement and validation.
    public static func pair(context ctx: MetalContext, repeats: Int = 9) async throws -> [Row] {
        let (a, b) = try testFrames(ctx)
        var rows: [Row] = []
        for path in [CorrelationPath.auto, .batched] {
            let analyzer = try PIVAnalyzer(config: PIVConfig(), context: ctx, path: path)
            var times: [Double] = []
            for i in 0..<(repeats + 2) {
                let t0 = DispatchTime.now().uptimeNanoseconds
                _ = try await analyzer.analyze(a, b)
                if i >= 2 { times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6) }
            }
            times.sort()
            rows.append(Row(label: "pair 96→64→64, \(path.rawValue) (wall)", windows: 64, milliseconds: times[times.count / 2]))
        }
        return rows
    }
}

/// SplitMix64, for repeatable test images.
struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
