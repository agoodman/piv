import Foundation

/// What a running analysis reports, in order: progress and previews while it runs, the
/// fields once analysis (and smoothing) is done, then the movie when it has been rendered.
public enum AnalysisEvent: Sendable {
    case progress(RunProgress)
    case preview(PreviewFrame)
    /// Fields ready; the movie may still be rendering.
    case analysed(RunResult)
    /// Everything done. `movie` is set when rendering was requested.
    case finished(RunResult, movie: URL?)
}

/// Entry point for apps: probes clips and runs analyses off the caller's thread, reporting
/// through an event stream. Cancelling the task that consumes the stream (or dropping the
/// stream) cancels the run: decoding stops, no further GPU work is submitted, and a partial
/// movie is deleted.
public actor PIVSession {
    public let context: MetalContext

    public init(context: MetalContext? = nil) throws {
        self.context = try context ?? MetalContext()
    }

    /// Size, frame count, container and camera capture rates of a clip.
    public static func probe(_ url: URL) async throws -> MediaInfo {
        try await MediaInfo.probe(url)
    }

    /// Runs `request`, then renders its quiver movie if `render` is given.
    public nonisolated func analyze(_ request: AnalysisRequest, render: RenderOptions? = nil) -> AsyncThrowingStream<AnalysisEvent, Error> {
        let ctx = context
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let result = try await analyzeRun(
                        request, context: ctx,
                        progress: { continuation.yield(.progress($0)) },
                        preview: { continuation.yield(.preview($0)) })
                    continuation.yield(.analysed(result))
                    var movie: URL?
                    if let render {
                        movie = try await renderQuiverMovie(result, options: render, context: ctx) {
                            continuation.yield(.progress($0))
                        }
                    }
                    continuation.yield(.finished(result, movie: movie))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
