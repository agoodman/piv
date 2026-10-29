import CoreGraphics
import Foundation

/// Everything that defines an analysis run. Defaults match `piv analyze`.
public struct AnalysisRequest: Sendable {
    public var source: URL
    /// Capture frame rate; `nil` uses camera metadata, then the container rate.
    public var captureRate: FrameRate?
    /// Frames to use (end exclusive); `nil` = the whole clip.
    public var frameRange: Range<Int>?
    /// Frame i is paired with i + stride.
    public var stride = 1
    /// A pair starts every `pairStep` frames.
    public var pairStep = 1
    public var roi: ROI?
    public var channel = Channel.luma
    public var background: BackgroundSpec? = BackgroundSpec()
    public var preprocess = PreprocessConfig()
    public var piv: PIVConfig = {
        var c = PIVConfig()
        c.mask = AnalysisRequest.defaultMask
        return c
    }()
    public var scale = ScaleSpec()
    public var sheetThicknessMM: Double?
    public var density = airDensity
    public var temporalSmoothing: TemporalSmoothing?
    public var path = CorrelationPath.auto
    /// Pairs analysed concurrently (keeps the GPU busy while the CPU validates).
    public var pairsInFlight = 3

    public init(source: URL) { self.source = source }

    /// Signal-mask defaults for the stored Y plane: the Rust tool's 0.04 / 0.015 scaled by 1.6.
    /// Those were tuned on FFmpeg's gray conversion, which on the sample clips is ~1.6× darker
    /// than Y. On C0141 these give the same no-signal fraction as `piv analyze` (63 vs 63 %).
    public static let defaultMask = MaskConfig(minMean: 0.064, minStd: 0.024)
}

/// One analysed pair's frames and capture time (pair midpoint, s from frame 0).
public struct PairTiming: Hashable, Sendable {
    public var frameA: Int
    public var frameB: Int
    public var time: Double
}

/// The result of a run: raw (validated) fields, and smoothed fields when requested.
public struct RunResult: Sendable {
    public var request: AnalysisRequest
    public var info: MediaInfo
    public var capture: CaptureRate
    public var frames: Range<Int>
    /// Seconds between the frames of a pair.
    public var dt: Double
    public var grid: Grid
    public var roi: ROI
    public var scale: Scale?
    public var energyUnits: EnergyUnits
    public var pairs: [PairTiming]
    public var raw: [VectorField]
    public var summaries: [ValidationSummary]
    /// Per-pair statistics of the raw fields (px/s units).
    public var rawStats: [FlowStats]
    public var smoothed: [VectorField]?
    public var support: [[UInt16]]?
    public var smoothedStats: [FlowStats]?
    /// Wall-clock seconds for the analysis loop (excluding the background estimate).
    public var elapsed: Double

    /// The fields the result is about: smoothed when smoothing was requested.
    public var fields: [VectorField] { smoothed ?? raw }
    public var stats: [FlowStats] { smoothedStats ?? rawStats }
}

public enum RunStage: String, Sendable { case background, analysing, smoothing, rendering }

public struct RunProgress: Sendable {
    public var stage: RunStage
    public var done: Int
    public var total: Int
}

/// Frame roles and pair list for a run: the same rules as `piv analyze`.
struct PairPlan {
    let start: Int, end: Int, stride: Int, pairStep: Int

    var pairStarts: [Int] { Array(Swift.stride(from: start, to: end - stride, by: pairStep)) }
    func isA(_ f: Int) -> Bool { f >= start && f + stride < end && (f - start) % pairStep == 0 }
    func isB(_ f: Int) -> Bool { f >= start + stride && isA(f - stride) }
}

/// Runs a whole analysis: probe, background, decode → prepare → correlate → validate for every
/// pair, then optional temporal smoothing.
public func analyzeRun(
    _ req: AnalysisRequest, context ctx: MetalContext,
    progress: (@Sendable (RunProgress) -> Void)? = nil,
    preview: (@Sendable (PreviewFrame) -> Void)? = nil
) async throws -> RunResult {
    let info = try await MediaInfo.probe(req.source)
    let capture = CaptureRate.resolve(override: req.captureRate, camera: info.cameraRate, container: info.containerRate)
    let start = req.frameRange?.lowerBound ?? 0
    let end = min(req.frameRange?.upperBound ?? info.frameCount, info.frameCount)
    guard req.stride >= 1, req.pairStep >= 1 else { throw PIVError.config("stride and pair step must be ≥ 1") }
    guard start + req.stride < end else {
        throw PIVError.config("frame range [\(start), \(end)) is too short for stride \(req.stride) (clip has \(info.frameCount) frames)")
    }
    try req.temporalSmoothing?.validate()
    let plan = PairPlan(start: start, end: end, stride: req.stride, pairStep: req.pairStep)
    let pairStarts = plan.pairStarts
    let dt = capture.rate.dt(stride: req.stride)
    let roi = req.roi ?? ROI(x: 0, y: 0, width: info.width, height: info.height)
    guard roi.x + roi.width <= info.width, roi.y + roi.height <= info.height else {
        throw PIVError.config("ROI \(roi.x),\(roi.y),\(roi.width),\(roi.height) exceeds the \(info.width)×\(info.height) frame")
    }
    // The FOV spans the full frame, not the ROI.
    let scale = req.scale.resolve(width: info.width, height: info.height)

    let preparer = try FramePreparer(context: ctx, config: req.preprocess, roi: roi, channel: req.channel)
    if let bg = req.background {
        progress?(RunProgress(stage: .background, done: 0, total: min(bg.frames, info.frameCount)))
        try await preparer.estimateBackground(bg, info: info) { done, total in
            progress?(RunProgress(stage: .background, done: done, total: total))
        }
    }

    let k = max(req.pairsInFlight, 1)
    var idle = try (0..<k).map { _ in try PIVAnalyzer(config: req.piv, context: ctx, path: req.path) }
    let grid = try idle[0].outputGrid(height: roi.height, width: roi.width)
    let decoder = try await VideoDecoder(info: info, range: start..<end, channel: req.channel)
    defer { decoder.cancel() }
    let t0 = Date()

    let previews = try preview.map { _ in try PreviewMaker(context: ctx) }

    struct Done: Sendable {
        let pair: Int
        let analyzer: PIVAnalyzer
        let result: PairResult
        let stats: FlowStats
    }
    var results = [PairResult?](repeating: nil, count: pairStarts.count)
    var rawStats = [FlowStats?](repeating: nil, count: pairStarts.count)
    var completed = 0

    try await withTaskCancellationHandler {
        try await withThrowingTaskGroup(of: Done.self) { group in
            func collect(_ d: Done) {
                results[d.pair] = d.result
                rawStats[d.pair] = d.stats
                idle.append(d.analyzer)
                completed += 1
                progress?(RunProgress(stage: .analysing, done: completed, total: pairStarts.count))
            }
            var held: [Int: PreparedFrame] = [:]
            progress?(RunProgress(stage: .analysing, done: 0, total: pairStarts.count))
            let profile = ProcessInfo.processInfo.environment["PIV_PROFILE"] != nil
            var (tDecode, tPrepare, tWait) = (0.0, 0.0, 0.0)
            var tMark = Date()
            func lap(_ acc: inout Double) { let now = Date(); acc += now.timeIntervalSince(tMark); tMark = now }
            defer {
                if profile {
                    print(String(format: "profile: waiting for decode %.2f s, prepare %.2f s, pair results %.2f s", tDecode, tPrepare, tWait))
                }
            }
            while let frame = try await decoder.nextFrame() {
                lap(&tDecode)
                try Task.checkCancellation()
                let f = frame.index
                let (aRole, bRole) = (plan.isA(f), plan.isB(f))
                guard aRole || bRole else { continue }
                held[f] = try preparer.prepare(frame)
                lap(&tPrepare)
                if bRole {
                    guard let a = held[f - req.stride], let b = held[f] else {
                        throw PIVError.io("frame \(f - req.stride) missing from the decoded stream")
                    }
                    while idle.isEmpty, let d = try await group.next() { collect(d) }
                    lap(&tWait)
                    let analyzer = idle.removeLast()
                    let pair = (f - req.stride - start) / req.pairStep
                    group.addTask {
                        let r = try await analyzer.analyze(a.gpu, b.gpu)
                        if let previews, let preview, previews.due() {
                            if let img = try? await previews.render(raw: a, field: r.field) {
                                preview(PreviewFrame(pair: pair, image: img))
                            }
                        }
                        return Done(pair: pair, analyzer: analyzer, result: r, stats: pairStats(r.field, dt: dt))
                    }
                }
                // Drop frames no later pair needs.
                held = held.filter { $0.key >= f + 1 - req.stride }
            }
            while let d = try await group.next() { collect(d) }
            lap(&tWait)
            try preparer.checkFailure()
        }
    } onCancel: {
        decoder.cancel()
    }
    let elapsed = Date().timeIntervalSince(t0)
    let raw = results.compactMap { $0 }, stats = rawStats.compactMap { $0 }
    guard completed == pairStarts.count, raw.count == pairStarts.count, stats.count == pairStarts.count else {
        throw PIVError.io("analysed \(completed) pairs, expected \(pairStarts.count) — the clip may be shorter than reported")
    }

    let pairs = pairStarts.map { a in
        PairTiming(frameA: a, frameB: a + req.stride, time: capture.rate.time(ofFrame: Double(a) + Double(req.stride) / 2))
    }
    var result = RunResult(
        request: req, info: info, capture: capture, frames: start..<end, dt: dt, grid: grid, roi: roi, scale: scale,
        energyUnits: EnergyUnits(metresPerPixel: scale?.metresPerPixel, thicknessMetres: req.sheetThicknessMM.map { $0 * 1e-3 }, density: req.density),
        pairs: pairs, raw: raw.map(\.field), summaries: raw.map(\.summary), rawStats: stats,
        smoothed: nil, support: nil, smoothedStats: nil, elapsed: elapsed
    )
    if let ts = req.temporalSmoothing {
        progress?(RunProgress(stage: .smoothing, done: 0, total: 1))
        let (fields, support) = ts.apply(to: result.raw)
        result.smoothed = fields
        result.support = support
        // Smoothed vectors are usable wherever they are defined (as in `piv smooth`).
        result.smoothedStats = fields.map { f in
            let k = Float(1 / dt)
            return flowStats(u: f.dx.map { $0 * k }, v: f.dy.map { $0 * k }, usable: f.dx.map(\.isFinite),
                             nx: f.grid.nx, ny: f.grid.ny, step: Float(f.grid.step))
        }
        progress?(RunProgress(stage: .smoothing, done: 1, total: 1))
    }
    return result
}

/// A quiver image of one pair, rendered while a run is in progress.
public struct PreviewFrame: @unchecked Sendable {
    public let pair: Int
    public let image: CGImage
}

/// Renders at most a couple of preview frames per second during a run. The style is fixed
/// from the first preview.
final class PreviewMaker: @unchecked Sendable {
    private let renderer: QuiverRenderer
    private let lock = NSLock()
    private var last = Date.distantPast
    private var style: QuiverStyle?
    static let interval = 0.5

    init(context: MetalContext) throws { renderer = try QuiverRenderer(context: context) }

    /// Whether a preview is due now (and claims it).
    func due() -> Bool {
        lock.withLock {
            let now = Date()
            guard now.timeIntervalSince(last) >= Self.interval else { return false }
            last = now
            return true
        }
    }

    /// Renders `field` over frame A. Called after the pair's command buffer completed, so the
    /// frame's upload (queued before it) has too.
    func render(raw: PreparedFrame, field: VectorField) async throws -> CGImage {
        let s = lock.withLock { style } ?? {
            let s = QuiverStyle.auto(raw: raw.raw.floats(raw.gpu.width * raw.gpu.height), field: field)
            lock.withLock { style = s }
            return s
        }()
        return try await renderer.image(raw: raw.raw, width: raw.gpu.width, height: raw.gpu.height, field: field, style: s)
    }
}
