import ArgumentParser
import CoreVideo
import Foundation
import PIVKit

@main
struct PIVKitCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pivkit",
        abstract: "Planar image velocimetry on the GPU (PIVKit).",
        subcommands: [Info.self, Analyze.self, Frame.self, Bench.self]
    )
}

extension FrameRate: ExpressibleByArgument {
    public init?(argument: String) { try? self.init(parsing: argument) }
}

extension ROI: ExpressibleByArgument {
    public init?(argument: String) { try? self.init(parsing: argument) }
}

extension Channel: ExpressibleByArgument {}
extension FOVAxis: ExpressibleByArgument {}
extension CorrelationPath: ExpressibleByArgument {}
extension TemporalSmoothing.Kernel: ExpressibleByArgument {}
extension MovieWriter.Codec: ExpressibleByArgument {}

struct Info: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "What's in a video file, and what rate was it captured at?")

    @Argument(help: "Input video.") var input: String

    func run() async throws {
        let info = try await MediaInfo.probe(URL(fileURLWithPath: input))
        let capture = CaptureRate.resolve(override: nil, camera: info.cameraRate, container: info.containerRate)
        print("file:       \(info.url.path) (\(info.fileSize / 1_000_000) MB)")
        print("video:      \(info.codec) \(info.width)×\(info.height), \(info.bitDepth)-bit, \(info.fullRange ? "full" : "video") range")
        print("frames:     \(info.frameCount) at \(info.containerRate) (container), \(String(format: "%.2f", info.playbackDuration)) s playback")
        if let c = info.cameraRate?.camera { print("camera:     \(c)") }
        print("capture:    \(capture.rate) — \(capture.detail)")
        print("            \(String(format: "%.3f", capture.rate.time(ofFrame: Double(info.frameCount)))) s of capture time")
        if let w = capture.warning { print("warning:    \(w)") }
    }
}

struct Analyze: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Analyse a clip into displacement and velocity fields (.npy directory).")

    @Argument(help: "Input video.") var input: String
    @Option(name: .shortAndLong, help: "Output directory (.npy files and run.json).") var output: String
    @Flag(help: "Overwrite existing output.") var force = false

    @Option(help: "First frame index.") var start: Int?
    @Option(help: "Frame index to stop before (exclusive).") var end: Int?
    @Option(help: "First frame, as capture-time seconds.") var startS: Double?
    @Option(help: "End, as capture-time seconds (exclusive).") var endS: Double?
    @Option(help: "Frame separation within a pair.") var stride = 1
    @Option(help: "Spacing between successive pairs.") var pairStep = 1
    @Option(help: "Capture frame rate override (e.g. 240, 1000, 239.76, 240000/1001).") var captureFps: FrameRate?

    @Option(help: "Image plane: luma (the stored Y plane) or green.") var channel = Channel.luma
    @Option(help: "Region of interest, x,y,w,h in full-frame pixels.") var roi: ROI?
    @Option(help: "Static background: none, min:N or mean:N.") var background = "min:40"
    @Option(help: "Gaussian smoothing sigma in px (0 = off).") var smooth: Float = PreprocessConfig().smoothSigma
    @Option(help: "Local contrast normalization radius in px (0 = off).") var normalizeRadius = PreprocessConfig().normalizeRadius

    @Option(parsing: .upToNextOption, help: "Window sizes per pass, largest first (e.g. --window 96 64).")
    var window: [Int] = [96, 64]
    @Option(help: "Window overlap fraction.") var overlap: Float = 0.5
    @Option(help: "Passes at the final window size.") var finalIterations = 2
    @Option(help: "Minimum window mean intensity after background subtraction (0–1). The Rust tool's 0.04 × 1.6, since the Y plane is brighter than its luma conversion.")
    var minMean: Float = AnalysisRequest.defaultMask.minMean
    @Option(help: "Minimum window intensity standard deviation (0–1); the Rust tool's 0.015 × 1.6.")
    var minStd: Float = AnalysisRequest.defaultMask.minStd
    @Option(help: "Correlation kernels: auto, fused, batched, mpsGraph.") var path = CorrelationPath.auto
    @Option(help: "Pairs analysed concurrently.") var pairsInFlight = 3

    @Option(help: "Minimum primary/secondary peak ratio.") var ppr: Float = ValidationConfig().pprThreshold
    @Option(help: "Normalized median test threshold.") var medianThreshold: Float = ValidationConfig().medianThreshold
    @Option(help: "Normalized median test noise level (px).") var medianEps: Float = ValidationConfig().medianEps
    @Flag(help: "Keep rejected vectors as NaN instead of replacing them.") var noReplace = false

    @Option(help: "Temporal smoothing over N pairs (odd).") var temporal: Int?
    @Option(help: "Temporal kernel: median or mean.") var temporalKernel = TemporalSmoothing.Kernel.median
    @Option(help: "Raw vectors needed in the temporal window (default ⌈N/3⌉).") var minSupport: Int?

    @Option(help: "Measured scale in pixels per millimetre.") var pxPerMm: Double?
    @Option(help: "Lens field of view in degrees (needs --sheet-distance).") var fovDeg: Double?
    @Option(help: "Which frame dimension --fov-deg spans.") var fovAxis = FOVAxis.diagonal
    @Option(help: "Camera-to-light-sheet distance in metres.") var sheetDistance: Double?
    @Option(help: "Light-sheet thickness in mm.") var sheetThicknessMm: Double?
    @Option(help: "Fluid density in kg/m³.") var density = 1.204

    @Flag(help: "Also write a quiver movie, <output>_quiver.mov (smoothed fields when --temporal is set).")
    var video = false
    @Option(help: "Playback rate of the quiver movie.") var videoFps = 30.0
    @Option(help: "Arrow length in image px per px of displacement (default: auto from the first pair).")
    var quiverScale: Float?
    @Option(help: "Draw every n-th vector.") var quiverEvery = 1
    @Option(help: "Movie codec: h264 or hevc.") var codec = MovieWriter.Codec.h264

    func run() async throws {
        let src = URL(fileURLWithPath: input)
        let out = URL(fileURLWithPath: output)
        if FileManager.default.fileExists(atPath: out.path) {
            guard force else { throw ValidationError("\(output) exists; pass --force to overwrite") }
            try FileManager.default.removeItem(at: out)
        }
        let info = try await MediaInfo.probe(src)
        let capture = CaptureRate.resolve(override: captureFps, camera: info.cameraRate, container: info.containerRate)
        if let w = capture.warning { FileHandle.standardError.write(Data("warning: \(w)\n".utf8)) }
        let toFrame = { (s: Double) in max(Int((s * capture.rate.fps).rounded()), 0) }
        let first = start ?? startS.map(toFrame) ?? 0
        let last = min(end ?? endS.map(toFrame) ?? info.frameCount, info.frameCount)

        var req = AnalysisRequest(source: src)
        req.captureRate = captureFps
        req.frameRange = first..<max(last, first)
        req.stride = stride
        req.pairStep = pairStep
        req.roi = roi
        req.channel = channel
        req.background = try BackgroundSpec.parse(background)
        req.preprocess.smoothSigma = smooth
        req.preprocess.normalizeRadius = normalizeRadius
        req.piv.windows = window
        req.piv.overlap = overlap
        req.piv.finalIterations = finalIterations
        req.piv.mask = MaskConfig(minMean: minMean, minStd: minStd)
        req.piv.validation.pprThreshold = ppr
        req.piv.validation.medianThreshold = medianThreshold
        req.piv.validation.medianEps = medianEps
        req.piv.validation.replace = !noReplace
        req.path = path
        req.pairsInFlight = pairsInFlight
        req.scale = ScaleSpec(pxPerMM: pxPerMm, fovDegrees: fovDeg, fovAxis: fovAxis, sheetDistanceMetres: sheetDistance)
        req.sheetThicknessMM = sheetThicknessMm
        req.density = density
        req.temporalSmoothing = temporal.map { TemporalSmoothing(window: $0, kernel: temporalKernel, minSupport: minSupport) }

        let pairCount = Array(Swift.stride(from: first, to: last - stride, by: pairStep)).count
        print("\(info.url.lastPathComponent): \(info.width)×\(info.height), capture \(capture.rate) (\(capture.detail))")
        print("frames [\(first), \(last)), stride \(stride) (dt \(String(format: "%.4g", capture.rate.dt(stride: stride) * 1e3)) ms), \(pairCount) pairs")

        let ctx = try MetalContext()
        let interactive = isatty(STDERR_FILENO) != 0
        let r = try await analyzeRun(req, context: ctx) { p in
            guard interactive else { return }
            let line = switch p.stage {
            case .background: "  background \(p.done)/\(p.total)"
            case .analysing: "  \(p.done)/\(p.total) pairs"
            case .smoothing: "  smoothing"
            case .rendering: "  rendering \(p.done)/\(p.total)"
            }
            FileHandle.standardError.write(Data("\r\(line)          ".utf8))
        }
        if interactive { FileHandle.standardError.write(Data("\r\(String(repeating: " ", count: 60))\r".utf8)) }
        try r.writeNpy(to: out)
        print(String(format: "analysed %d pairs in %.1f s (%.1f pairs/s) → %@", r.pairs.count, r.elapsed, Double(r.pairs.count) / r.elapsed, out.path))
        summarize(r)
        if video {
            var opts = RenderOptions()
            opts.url = out.deletingLastPathComponent().appendingPathComponent(out.lastPathComponent + "_quiver.mov")
            opts.fps = videoFps
            opts.arrowScale = quiverScale
            opts.every = quiverEvery
            opts.codec = codec
            let t0 = Date()
            let movie = try await renderQuiverMovie(r, options: opts, context: ctx) { p in
                guard interactive else { return }
                FileHandle.standardError.write(Data("\r  rendering \(p.done)/\(p.total)          ".utf8))
            }
            if interactive { FileHandle.standardError.write(Data("\r\(String(repeating: " ", count: 60))\r".utf8)) }
            print(String(format: "movie: %@ (%.1f s, %.0f frames/s)", movie.path, Date().timeIntervalSince(t0),
                         Double(r.pairs.count) / Date().timeIntervalSince(t0)))
        }

    }

    func summarize(_ r: RunResult) {
        var (vectors, noSignal, rejected, replaced) = (0, 0, 0, 0)
        var mags: [Float] = []
        for f in r.raw {
            for k in f.flags.indices {
                vectors += 1
                let fl = f.flags[k]
                if fl.contains(.noSignal) { noSignal += 1 }
                else if !fl.isValid { rejected += 1; if fl.contains(.replaced) { replaced += 1 } }
                else if vectors % 3 == 0 { mags.append(hypotf(f.dx[k], f.dy[k])) }
            }
        }
        mags.sort()
        func pct(_ q: Double) -> Float { mags.isEmpty ? .nan : mags[Int((Double(mags.count - 1) * q).rounded())] }
        let n = Double(max(vectors, 1))
        print(String(format: "vectors: %.1f%% valid, %.1f%% rejected (%.1f%% replaced), %.1f%% no signal",
                     100 * Double(vectors - noSignal - rejected) / n, 100 * Double(rejected) / n, 100 * Double(replaced) / n, 100 * Double(noSignal) / n))
        print(String(format: "|d| (px/pair): median %.2f, p95 %.2f, p99 %.2f", pct(0.5), pct(0.95), pct(0.99)))
    }
}

struct Frame: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Decode one frame's luma (stored Y plane) as raw 8-bit values, for checking against FFmpeg.")

    @Argument(help: "Input video.") var input: String
    @Argument(help: "Frame index.") var index: Int
    @Option(name: .shortAndLong, help: "Output file (width × height bytes).") var output: String
    @Option(help: "Instead, print a checksum of each of this many frames from the index on, decoded in sequence.")
    var checksums: Int?

    func run() async throws {
        let info = try await MediaInfo.probe(URL(fileURLWithPath: input))
        if let n = checksums {
            let d = try await VideoDecoder(info: info, range: index..<(index + n), channel: .luma)
            while let f = try await d.nextFrame() {
                let pb = f.pixelBuffer
                CVPixelBufferLockBaseAddress(pb, .readOnly)
                let (w, h) = (CVPixelBufferGetWidthOfPlane(pb, 0), CVPixelBufferGetHeightOfPlane(pb, 0))
                let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
                let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self)
                var sum: UInt64 = 0
                for y in 0..<h { for x in 0..<w { sum &+= UInt64(base[y * stride + x]) * UInt64((x * 7 + y * 13) % 251 + 1) } }
                CVPixelBufferUnlockBaseAddress(pb, .readOnly)
                print(f.index, sum)
            }
            return
        }
        let f = try await VideoDecoder.frame(index, info: info, channel: .luma)
        let pb = f.pixelBuffer
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let (w, h) = (CVPixelBufferGetWidthOfPlane(pb, 0), CVPixelBufferGetHeightOfPlane(pb, 0))
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self)
        var out = Data(capacity: w * h)
        for y in 0..<h { out.append(base + y * stride, count: w) }
        try out.write(to: URL(fileURLWithPath: output))
        print("frame \(f.index): \(w)×\(h) → \(output)")
    }
}

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Time the GPU correlation kernels on a synthetic 1920×1080 pair.")

    @Option(help: "Timed repeats per measurement (median reported).")
    var repeats = 9

    func run() async throws {
        let ctx = try MetalContext()
        print("device: \(ctx.device.name), max threadgroup memory \(ctx.device.maxThreadgroupMemoryLength) B")
        print("")
        print("window  kernels                                  GPU ms")
        for r in try await CorrelationBenchmark.correlation(context: ctx, repeats: repeats) {
            print(String(format: "%4d px  %-40@ %7.3f", r.windows, r.label as NSString, r.milliseconds))
        }
        print("")
        for r in try await CorrelationBenchmark.prepare(context: ctx, repeats: repeats) {
            print(String(format: "%-48@ %7.3f ms", r.label as NSString, r.milliseconds))
        }
        print("")
        for r in try await CorrelationBenchmark.pair(context: ctx, repeats: repeats) {
            print(String(format: "%-48@ %7.3f ms  (%.0f pairs/s)", r.label as NSString, r.milliseconds, 1000 / r.milliseconds))
        }
    }
}
