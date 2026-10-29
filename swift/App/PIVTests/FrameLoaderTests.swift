import CoreGraphics
import Foundation
import Testing
@testable import PIV
import PIVKit

/// Scrubbing against a clip in the app's container (the sandbox can read it there); skipped
/// when it isn't present.
private let excerpt = FileManager.default.temporaryDirectory.appendingPathComponent("c0141_excerpt.mp4")

@MainActor
@Suite struct FrameLoaderTests {
    @Test(.enabled(if: FileManager.default.fileExists(atPath: excerpt.path), "needs c0141_excerpt.mp4 in the container's tmp"))
    func keepsUpWithADragAndSettlesOnTheExactFrame() async throws {
        let info = try await MediaInfo.probe(excerpt)
        let loader = FrameLoader(info: info)
        // A drag across the clip: a new position every 8 ms (~120 Hz) for ~1.2 s.
        var updates = 0
        var last: CGImage?
        let t0 = Date()
        for f in stride(from: 0, to: info.frameCount, by: 1) {
            loader.request(f, exact: false)
            try await Task.sleep(nanoseconds: 8_000_000)
            if let img = loader.image, img !== last { updates += 1; last = img }
        }
        let drag = Date().timeIntervalSince(t0)
        // Release: settle on the exact frame.
        let target = info.frameCount - 1
        let t1 = Date()
        loader.request(target, exact: true)
        while loader.exactFrame != target && Date().timeIntervalSince(t1) < 2 {
            try await Task.sleep(nanoseconds: 2_000_000)
        }
        let settle = Date().timeIntervalSince(t1)
        print(String(format: "scrub: %d preview updates in %.2f s drag (%.0f/s); exact frame %.0f ms after release",
                     updates, drag, Double(updates) / drag, settle * 1000))
        #expect(loader.exactFrame == target)
        #expect(Double(updates) / drag > 15, "preview should update continuously while dragging")
        #expect(settle < 0.3)
    }
}
