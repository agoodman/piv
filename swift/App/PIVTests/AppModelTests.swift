import Foundation
import Testing
@testable import PIV
import PIVKit

/// Scripted stand-in for PIVSession.
final class FakeService: AnalysisService, @unchecked Sendable {
    enum Script { case succeed, fail, hang }
    let info: MediaInfo
    let script: Script
    private(set) var requests: [AnalysisRequest] = []

    init(script: Script = .succeed, camera: Bool = true) {
        let url = URL(fileURLWithPath: "/tmp/fake-\(UUID().uuidString).mp4")
        info = MediaInfo(url: url, fileSize: 1, codec: "avc1", width: 256, height: 128, bitDepth: 8, fullRange: false,
                         frameCount: 100, containerRate: try! FrameRate(50), startTime: 0,
                         cameraRate: camera ? CameraRate(rate: try! FrameRate(1000), source: "Sony NonRealTimeMeta", raw: "1000p") : nil)
        self.script = script
    }

    func probe(_ url: URL) async throws -> MediaInfo { info }

    func analyze(_ request: AnalysisRequest, render: RenderOptions?) -> AsyncThrowingStream<AnalysisEvent, Error> {
        requests.append(request)
        let script = script, info = info
        return AsyncThrowingStream { c in
            c.yield(.progress(RunProgress(stage: .analysing, done: 1, total: 2)))
            switch script {
            case .succeed:
                let r = Self.result(request, info)
                c.yield(.analysed(r))
                c.yield(.finished(r, movie: URL(fileURLWithPath: "/tmp/fake-movie.mov")))
                c.finish()
            case .fail:
                c.finish(throwing: PIVError.io("decoder exploded"))
            case .hang:
                break // until cancelled
            }
        }
    }

    static func result(_ req: AnalysisRequest, _ info: MediaInfo) -> RunResult {
        let grid = try! Grid(height: 128, width: 256, window: 64, overlap: 0.5)
        let f = VectorField(grid: grid, dx: .init(repeating: 1, count: grid.count), dy: .init(repeating: 0, count: grid.count),
                            peakRatio: .init(repeating: 3, count: grid.count), flags: .init(repeating: [], count: grid.count))
        let stats = flowStats(u: f.dx, v: f.dy, usable: .init(repeating: true, count: grid.count), nx: grid.nx, ny: grid.ny, step: 32)
        return RunResult(
            request: req, info: info,
            capture: CaptureRate.resolve(override: req.captureRate, camera: info.cameraRate, container: info.containerRate),
            frames: 0..<10, dt: 0.001, grid: grid, roi: ROI(x: 0, y: 0, width: 256, height: 128), scale: nil,
            energyUnits: EnergyUnits(metresPerPixel: nil, thicknessMetres: nil), pairs: [PairTiming(frameA: 0, frameB: 1, time: 0.0005)],
            raw: [f], summaries: [ValidationSummary()], rawStats: [stats], elapsed: 0.1)
    }
}

@MainActor
@Suite struct AppModelTests {
    @Test func openingAClipPrefillsSettingsFromMetadata() async {
        let fake = FakeService()
        let m = AppModel(service: fake)
        await m.open(fake.info.url)
        guard case .setup = m.phase else { Issue.record("expected setup, got \(m.phase)"); return }
        #expect(m.settings?.captureFPS == "1000")
        #expect(m.settings?.endFrame == 100)
        #expect(m.captureRateNote?.contains("Sony") == true)
    }

    @Test func withoutMetadataTheNoteWarnsAboutTheContainerRate() async {
        let fake = FakeService(camera: false)
        let m = AppModel(service: fake)
        await m.open(fake.info.url)
        #expect(m.settings?.captureFPS == "50")
        #expect(m.captureRateNote?.contains("container rate") == true)
    }

    @Test func invalidSettingsShowAnErrorAndDontRun() async {
        let fake = FakeService()
        let m = AppModel(service: fake)
        await m.open(fake.info.url)
        m.settings?.captureFPS = "fast"
        m.run()
        #expect(m.error?.contains("isn't a rate") == true)
        #expect(!m.isRunning && fake.requests.isEmpty)
    }

    @Test func aSuccessfulRunEndsInTheResult() async throws {
        let fake = FakeService()
        let m = AppModel(service: fake)
        await m.open(fake.info.url)
        m.settings?.stride = 5
        m.settings?.smoothing = true
        m.run()
        #expect(m.isRunning)
        try await waitUntil { if case .result = m.phase { true } else { false } }
        guard case .result(let out) = m.phase else { return }
        #expect(out.movie.lastPathComponent == "fake-movie.mov" && out.fps == 30)
        let req = try #require(fake.requests.first)
        let k = try FrameRate(1000)
        #expect(req.stride == 5 && req.temporalSmoothing?.window == 9 && req.captureRate == k)
        #expect(req.piv.mask == AnalysisRequest.defaultMask)
    }

    @Test func aFailedRunReturnsToSetupWithTheError() async throws {
        let fake = FakeService(script: .fail)
        let m = AppModel(service: fake)
        await m.open(fake.info.url)
        m.run()
        try await waitUntil { m.error != nil }
        #expect(m.error?.contains("decoder exploded") == true)
        guard case .setup = m.phase else { Issue.record("expected setup"); return }
    }

    @Test func cancellingReturnsToSetup() async throws {
        let fake = FakeService(script: .hang)
        let m = AppModel(service: fake)
        await m.open(fake.info.url)
        m.run()
        try await waitUntil { if case .running(let s) = m.phase { s.total == 2 } else { false } }
        m.cancel()
        guard case .setup = m.phase else { Issue.record("expected setup"); return }
        #expect(m.error == nil)
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<200 {
            if condition() { return }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        Issue.record("timed out")
    }
}

@Suite struct RunSettingsTests {
    let info = FakeService().info

    @Test func mapsToTheRequest() throws {
        var s = RunSettings(for: info)
        s.startFrame = 10
        s.endFrame = 60
        s.stride = 5
        s.pairStep = 2
        s.roi = ROI(x: 0, y: 0, width: 128, height: 128)
        s.scaleMode = .fov
        s.distanceMM = 300
        s.windows = "64, 32"
        s.background = .none
        let req = try s.request(info: info).get()
        #expect(req.frameRange == 10..<60 && req.stride == 5 && req.pairStep == 2)
        #expect(req.piv.windows == [64, 32] && req.background == nil)
        #expect(req.scale.sheetDistanceMetres == 0.3 && req.scale.fovDegrees == 84)
        #expect(s.pairCount == 23) // starts 10, 12, …, 54
    }

    @Test func rejectsBadInput() {
        var s = RunSettings(for: info)
        s.windows = "80"
        #expect(throws: SettingsError.self) { try s.request(info: info).get() }
        s = RunSettings(for: info)
        s.roi = ROI(x: 200, y: 0, width: 128, height: 128)
        #expect(throws: SettingsError.self) { try s.request(info: info).get() }
        s = RunSettings(for: info)
        s.startFrame = 99
        #expect(throws: SettingsError.self) { try s.request(info: info).get() }
    }
}
