import AVFoundation
import Observation
import PIVKit

/// Decodes preview frames for scrubbing. One generator per clip, at most one request in flight:
/// when a request finishes, the loader goes straight to the latest requested frame, so the
/// preview keeps up with the slider instead of queueing (or debouncing) every position.
/// While dragging it accepts the nearest frame that decodes quickly (within ±0.5 s); when the
/// drag ends it shows the exact frame.
@MainActor @Observable
final class FrameLoader {
    private(set) var image: CGImage?
    /// The frame `image` shows, when it is exact.
    private(set) var exactFrame: Int?

    @ObservationIgnored private let info: MediaInfo
    @ObservationIgnored private let fast: GeneratorBox
    @ObservationIgnored private let exact: GeneratorBox
    @ObservationIgnored private var pending: (frame: Int, exact: Bool)?
    @ObservationIgnored private var loading = false

    init(info: MediaInfo) {
        self.info = info
        let asset = AVURLAsset(url: info.url)
        func generator(tolerance: CMTime) -> GeneratorBox {
            let g = AVAssetImageGenerator(asset: asset)
            g.requestedTimeToleranceBefore = tolerance
            g.requestedTimeToleranceAfter = tolerance
            g.maximumSize = CGSize(width: 1920, height: 1920)
            return GeneratorBox(g)
        }
        fast = generator(tolerance: CMTime(value: 1, timescale: 2))
        exact = generator(tolerance: .zero)
    }

    /// Asks for `frame`; superseded requests that haven't started are dropped.
    func request(_ frame: Int, exact: Bool) {
        if exact, exactFrame == frame { return }
        pending = (frame, exact)
        guard !loading else { return }
        loading = true
        Task { await pump() }
    }

    private func pump() async {
        while let next = pending {
            pending = nil
            let r = info.containerRate
            let t = CMTimeAdd(CMTime(seconds: info.startTime, preferredTimescale: 600_000),
                              CMTime(value: Int64(next.frame) * Int64(r.den), timescale: CMTimeScale(r.num)))
            if let img = try? await Self.decode(next.exact ? exact : fast, at: t) {
                image = img
                exactFrame = next.exact ? next.frame : nil
            }
        }
        loading = false
    }

    /// Off the main actor: the generator is used by one request at a time.
    nonisolated private static func decode(_ box: GeneratorBox, at t: CMTime) async throws -> CGImage {
        try await box.generator.image(at: t).image
    }
}

/// AVAssetImageGenerator isn't Sendable; the loader never runs two requests at once.
final class GeneratorBox: @unchecked Sendable {
    let generator: AVAssetImageGenerator
    init(_ generator: AVAssetImageGenerator) { self.generator = generator }
}
