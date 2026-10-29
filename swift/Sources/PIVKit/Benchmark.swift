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

    static func gpuMilliseconds(_ ctx: MetalContext, repeats: Int, _ encode: (MPSCommandBuffer) throws -> Void) async throws -> Double {
        var times: [Double] = []
        for i in 0..<(repeats + 2) {
            let cb = MPSCommandBuffer(from: ctx.queue)
            try encode(cb)
            let first = cb.rootCommandBuffer, last = cb.commandBuffer
            try await ctx.run(cb)
            if i >= 2 { times.append((last.gpuEndTime - first.gpuStartTime) * 1e3) } // skip warm-up
        }
        times.sort()
        return times[times.count / 2]
    }

    /// Per-pass correlation at 1080p with each kernel choice that supports the window size.
    public static func correlation(context ctx: MetalContext, repeats: Int = 9) async throws -> [Row] {
        let (a, b) = try testFrames(ctx)
        var rows: [Row] = []
        for n in [96, 64, 48, 32] {
            let grid = try Grid(height: a.height, width: a.width, window: n, overlap: 0.5)
            let origins = PIVAnalyzer.placeWindows(grid: grid, width: a.width, height: a.height, predictor: nil)
            let corr = Correlator(context: ctx)
            let auto = try corr.kernel(n, .auto)
            for path in [CorrelationPath.fused, .batched, .mpsGraph] {
                guard let kind = try? corr.kernel(n, path) else { continue }
                // CPU cost of encoding one pass (median), after a warm-up that builds graphs.
                var cpu: [Double] = []
                for i in 0..<12 {
                    let cb = MPSCommandBuffer(from: ctx.queue)
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    _ = try corr.encode(cb, a: a, b: b, n: n, origins: origins, mask: MaskConfig(), path: path)
                    if i >= 2 { cpu.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6) }
                    try await ctx.run(cb)
                }
                cpu.sort()
                rows.append(Row(label: "  \(path.rawValue) CPU encode", windows: n, milliseconds: cpu[cpu.count / 2]))
                let t = try await gpuMilliseconds(ctx, repeats: repeats) { cb in
                    _ = try corr.encode(cb, a: a, b: b, n: n, origins: origins, mask: MaskConfig(), path: path)
                }
                let chosen = kind == auto ? "  ← auto" : ""
                rows.append(Row(label: "\(path.rawValue) (\(origins.count) windows)\(chosen)", windows: n, milliseconds: t))
            }
        }
        return rows
    }

    /// Frame preparation (blur σ 1 + local normalization r 4, the defaults) at 1080p: wall time
    /// per frame, run alone.
    public static func prepare(context ctx: MetalContext, repeats: Int = 9) async throws -> [Row] {
        let (a, _) = try testFrames(ctx)
        let values = a.stats.floats(a.width * a.height)
        var rows: [Row] = []
        for (label, sigma, radius) in [("prepare: blur + normalize", Float(1), 4), ("prepare: blur only", Float(1), 0), ("prepare: normalize only", Float(0), 4)] {
            var c = PreprocessConfig()
            c.smoothSigma = sigma
            c.normalizeRadius = radius
            let p = try FramePreparer(context: ctx, config: c, roi: ROI(x: 0, y: 0, width: a.width, height: a.height), channel: .luma)
            var times: [Double] = []
            for i in 0..<(repeats + 2) {
                let t0 = DispatchTime.now().uptimeNanoseconds
                _ = try await p.prepare(raw: values)
                if i >= 2 { times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6) }
            }
            times.sort()
            rows.append(Row(label: label + " (wall)", windows: 0, milliseconds: times[times.count / 2]))
        }
        return rows
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
            rows.append(Row(label: "pair 96→64→64, \(path.rawValue) (wall, unpipelined)", windows: 64, milliseconds: times[times.count / 2]))
        }
        // Throughput with several pairs in flight on the same prepared frames: the ceiling for
        // a run once decoding and preparation are out of the way.
        for k in [2, 3, 4, 6] {
            let analyzers = try (0..<k).map { _ in try PIVAnalyzer(config: PIVConfig(), context: ctx) }
            let n = 120
            let t0 = DispatchTime.now().uptimeNanoseconds
            try await withThrowingTaskGroup(of: Void.self) { group in
                for (i, an) in analyzers.enumerated() {
                    group.addTask { for _ in stride(from: i, to: n, by: k) { _ = try await an.analyze(a, b) } }
                }
                try await group.waitForAll()
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / Double(n)
            rows.append(Row(label: "pair 96→64→64, auto, \(k) in flight (wall/pair)", windows: 64, milliseconds: ms))
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
