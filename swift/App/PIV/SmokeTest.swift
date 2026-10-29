#if DEBUG
import AppKit
import Foundation
import PIVKit

/// Debug-only end-to-end check, for running the app unattended: with `PIV_SMOKE_CLIP` set (a
/// clip the sandboxed app can read, e.g. in its container), opens the clip, runs a short
/// analysis, and saves window snapshots and a summary to `PIV_SMOKE_OUT`, then quits.
@MainActor
enum SmokeTest {
    static func runIfRequested(_ model: AppModel) {
        let env = ProcessInfo.processInfo.environment
        guard let clip = env["PIV_SMOKE_CLIP"], let out = env["PIV_SMOKE_OUT"] else { return }
        let outDir = URL(fileURLWithPath: out)
        Task {
            var log: [String] = []
            func note(_ s: String) { log.append(s) }
            try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
            try? await Task.sleep(nanoseconds: 800_000_000)
            await model.open(URL(fileURLWithPath: clip))
            note("opened: phase=\(model.phase) error=\(model.error ?? "-")")
            if let fps = env["PIV_SMOKE_FPS"] { model.settings?.captureFPS = fps }
            model.settings?.stride = Int(env["PIV_SMOKE_STRIDE"] ?? "5") ?? 5
            let end = min(model.settings?.endFrame ?? 0, Int(env["PIV_SMOKE_FRAMES"] ?? "150") ?? 150)
            model.settings?.endFrame = end
            model.settings?.smoothing = env["PIV_SMOKE_SMOOTH"] != nil
            note("note: \(model.captureRateNote ?? "-")")
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            snapshot("1-setup", outDir)
            model.run()
            var shotRunning = false
            for _ in 0..<600 {
                try? await Task.sleep(nanoseconds: 100_000_000)
                if case .running(let s) = model.phase, !shotRunning, s.stage == .analysing, s.done > 5 {
                    snapshot("2-running", outDir)
                    shotRunning = true
                }
                if case .running(let s) = model.phase, let img = s.preview,
                   !FileManager.default.fileExists(atPath: outDir.appendingPathComponent("2-preview.png").path) {
                    let rep = NSBitmapImageRep(cgImage: img)
                    try? rep.representation(using: .png, properties: [:])?.write(to: outDir.appendingPathComponent("2-preview.png"))
                    note("preview: pair image \(img.width)×\(img.height)")
                }
                if case .result = model.phase { break }
                if model.error != nil { break }
            }
            if case .result(let o) = model.phase {
                note("result: \(o.result.pairs.count) pairs, movie \(o.movie.path), valid \(o.summary.valid)")
                try? FileManager.default.copyItem(at: o.movie, to: outDir.appendingPathComponent("movie.mov"))
                try? await Task.sleep(nanoseconds: 2_500_000_000)
                snapshot("3-result", outDir)
            } else {
                note("no result: phase=\(model.phase) error=\(model.error ?? "-")")
                snapshot("3-error", outDir)
            }
            try? log.joined(separator: "\n").write(to: outDir.appendingPathComponent("smoke.txt"), atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
        }
    }

    static func snapshot(_ name: String, _ dir: URL) {
        guard let view = NSApp.windows.first(where: { $0.isVisible })?.contentView,
              let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: dir.appendingPathComponent("\(name).png"))
    }
}
#endif
