import AppKit
import Foundation
import Observation
import PIVKit

/// What the app needs from PIVKit, so the model can be tested with a fake.
protocol AnalysisService: Sendable {
    func probe(_ url: URL) async throws -> MediaInfo
    func analyze(_ request: AnalysisRequest, render: RenderOptions?) -> AsyncThrowingStream<AnalysisEvent, Error>
}

extension PIVSession: AnalysisService {
    nonisolated func probe(_ url: URL) async throws -> MediaInfo { try await PIVSession.probe(url) }
}

/// The app's state: one clip at a time, moving through setup → running → result.
@MainActor @Observable
final class AppModel {
    enum Phase {
        case empty
        case setup
        case running(RunStatus)
        case result(RunOutput)
    }

    struct RunStatus {
        var stage = RunStage.background
        var done = 0
        var total = 0
        var pairsPerSecond: Double?
        var eta: TimeInterval?
        var preview: CGImage?
        var started = Date()
    }

    struct RunOutput {
        var result: RunResult
        var movie: URL
        /// Movie frame rate (one frame per pair).
        var fps: Double
        var summary: RunSummary
    }

    private(set) var phase = Phase.empty
    private(set) var info: MediaInfo?
    var settings: RunSettings?
    /// A message to show (failed import, invalid settings, failed run).
    var error: String?

    private let service: AnalysisService
    private var runTask: Task<Void, Never>?
    private var analysisStart: Date?
    private var renderFPS = 30.0
    /// The last run's movie, in the temporary directory until the next run replaces it.
    private var lastMovie: URL?

    private func discardLastMovie() {
        if let m = lastMovie { try? FileManager.default.removeItem(at: m) }
        lastMovie = nil
    }

    init(service: AnalysisService) {
        self.service = service
    }

    var isRunning: Bool { if case .running = phase { true } else { false } }

    // MARK: Import

    func open(_ url: URL) async {
        guard !isRunning else { return }
        do {
            let info = try await service.probe(url)
            discardLastMovie()
            self.info = info
            settings = RunSettings.saved(for: info) ?? RunSettings(for: info)
            error = nil
            phase = .setup
        } catch {
            self.error = "Couldn't open \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }

    func resetSettings() {
        guard let info else { return }
        settings = RunSettings(for: info)
    }

    /// Where the capture rate comes from, and a warning when it may be wrong.
    var captureRateNote: String? {
        guard let info else { return nil }
        guard let rate = settings?.captureRate else { return "Not a frame rate: try 1000, 239.76 or 240000/1001." }
        if let c = info.cameraRate {
            return rate == c.rate ? "From \(c.source) metadata (\(c.raw))." : "Overrides \(c.rate) from \(c.source) metadata."
        }
        if rate == info.containerRate {
            return "No capture-rate metadata: this is the container rate. Set the real capture rate if the footage is slowed down."
        }
        return "Set by you (the container says \(info.containerRate))."
    }

    // MARK: Running

    func run() {
        guard let info, let settings, !isRunning else { return }
        let request: AnalysisRequest
        switch settings.request(info: info) {
        case .success(let r): request = r
        case .failure(let e):
            error = e.message
            return
        }
        settings.save(for: info)
        error = nil
        var render = RenderOptions()
        render.fps = settings.videoFPS
        renderFPS = settings.videoFPS
        discardLastMovie()
        phase = .running(RunStatus())
        analysisStart = nil
        let stream = service.analyze(request, render: render)
        runTask = Task { [weak self] in
            do {
                for try await event in stream {
                    self?.handle(event)
                }
                // Cancelled streams end without `.finished`.
                if let self, self.isRunning { self.phase = .setup }
            } catch is CancellationError {
                self?.phase = .setup
            } catch {
                self?.error = "The analysis failed: \(error.localizedDescription)"
                self?.phase = .setup
            }
        }
    }

    func handle(_ event: AnalysisEvent) {
        guard case .running(var s) = phase else { return }
        switch event {
        case .progress(let p):
            if p.stage == .analysing && analysisStart == nil { analysisStart = Date() }
            s.stage = p.stage
            s.done = p.done
            s.total = p.total
            if p.stage == .analysing, let t0 = analysisStart, p.done > 0 {
                let rate = Double(p.done) / Date().timeIntervalSince(t0)
                s.pairsPerSecond = rate
                s.eta = Double(p.total - p.done) / rate
            } else if p.stage != .analysing {
                s.pairsPerSecond = nil
                s.eta = nil
            }
            phase = .running(s)
        case .preview(let f):
            s.preview = f.image
            phase = .running(s)
        case .analysed:
            break
        case .finished(let result, let movie):
            guard let movie else {
                error = "The run finished without a movie."
                phase = .setup
                return
            }
            lastMovie = movie
            phase = .result(RunOutput(result: result, movie: movie, fps: renderFPS, summary: RunSummary(result)))
        }
    }

    func cancel() {
        runTask?.cancel()
        runTask = nil
        if isRunning { phase = .setup }
    }

    func backToSetup() {
        guard info != nil else { return }
        phase = .setup
    }
}
