import CoreVideo
import Foundation

/// How to render the quiver movie of a run.
public struct RenderOptions: Hashable, Sendable {
    /// Where to write the movie; `nil` = a new file in the temporary directory.
    public var url: URL?
    /// Playback frame rate.
    public var fps = 30.0
    public var codec = MovieWriter.Codec.h264
    /// Arrow length per px of displacement; `nil` = automatic from the first pair.
    public var arrowScale: Float?
    /// Draw every n-th vector in each direction.
    public var every = 1

    public init() {}
}

/// Renders the quiver movie of a finished run: frame A of every pair with that pair's field
/// (smoothed when the run was smoothed, raw otherwise), re-decoding the clip. Returns the
/// movie's URL.
public func renderQuiverMovie(
    _ r: RunResult, options: RenderOptions, context ctx: MetalContext,
    progress: (@Sendable (RunProgress) -> Void)? = nil
) async throws -> URL {
    let url = options.url ?? FileManager.default.temporaryDirectory.appendingPathComponent("pivkit-quiver-\(UUID().uuidString).mov")
    let fields = r.fields
    guard !fields.isEmpty else { throw PIVError.config("nothing to render: the run has no pairs") }
    let preparer = try FramePreparer(context: ctx, config: r.request.preprocess, roi: r.roi, channel: r.request.channel)
    let renderer = try QuiverRenderer(context: ctx)
    let writer = try MovieWriter(url: url, width: r.roi.width, height: r.roi.height, fps: options.fps, codec: options.codec)
    var finished = false
    defer {
        if !finished {
            writer.cancel()
            try? FileManager.default.removeItem(at: url)
        }
    }

    let starts = r.pairs.map(\.frameA)
    let wanted = Set(starts)
    let decoder = try await VideoDecoder(info: r.info, range: starts.first!..<(starts.last! + 1), channel: r.request.channel)
    defer { decoder.cancel() }

    var style: QuiverStyle?
    var next = 0
    progress?(RunProgress(stage: .rendering, done: 0, total: fields.count))
    while next < fields.count, let frame = try await decoder.nextFrame() {
        try Task.checkCancellation()
        guard wanted.contains(frame.index) else { continue }
        guard frame.index == starts[next] else {
            throw PIVError.io("frame \(starts[next]) missing from the decoded stream")
        }
        let raw = try preparer.uploadRaw(frame)
        if style == nil {
            // From the first pair, as in `piv analyze` (one read back, then fixed for the run).
            var s = QuiverStyle.auto(raw: try await read(raw, ctx), field: fields[0])
            if let a = options.arrowScale { s.arrowScale = a }
            s.every = options.every
            style = s
        }
        let pb = try writer.makePixelBuffer()
        try await renderer.render(into: pb, raw: raw.raw, width: r.roi.width, height: r.roi.height, field: fields[next], style: style!)
        try await writer.append(pb)
        next += 1
        progress?(RunProgress(stage: .rendering, done: next, total: fields.count))
    }
    guard next == fields.count else { throw PIVError.io("rendered \(next) of \(fields.count) frames") }
    try await writer.finish()
    finished = true
    return url
}

/// Waits for queued GPU work and copies a prepared frame's raw intensities out.
private func read(_ f: PreparedFrame, _ ctx: MetalContext) async throws -> [Float] {
    guard let cb = ctx.queue.makeCommandBuffer() else { throw PIVError.metal("no command buffer") }
    try await ctx.run(cb) // empty: completes after everything queued before it
    return f.raw.floats(f.gpu.width * f.gpu.height)
}
