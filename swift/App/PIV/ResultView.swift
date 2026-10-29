import AppKit
import AVFoundation
import AVKit
import Charts
import PIVKit
import SwiftUI

/// Playback state shared by the player, the readout and the charts: which pair is on screen.
@MainActor @Observable
final class PlaybackState {
    let player: AVPlayer
    let fps: Double
    let pairs: Int
    private(set) var pair = 0
    private var observer: Any?

    init(movie: URL, fps: Double, pairs: Int) {
        player = AVPlayer(url: movie)
        self.fps = fps
        self.pairs = pairs
        observer = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: CMTimeScale(max(fps, 1) * 2)), queue: .main) { [weak self] t in
            MainActor.assumeIsolated { self?.update(t) }
        }
    }

    private func update(_ t: CMTime) {
        guard t.isNumeric else { return }
        pair = min(max(Int(t.seconds * fps), 0), max(pairs - 1, 0))
    }

    func seek(toPair i: Int) {
        let i = min(max(i, 0), pairs - 1)
        player.seek(to: CMTime(seconds: (Double(i) + 0.5) / fps, preferredTimescale: 600_000),
                    toleranceBefore: .zero, toleranceAfter: .zero)
        pair = i
    }

    func stop() {
        player.pause()
        if let observer { player.removeTimeObserver(observer) }
        observer = nil
    }
}

struct ResultView: View {
    @Environment(AppModel.self) private var model
    let output: AppModel.RunOutput
    @State private var playback: PlaybackState?

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top, spacing: 0) {
                VStack(spacing: 8) {
                    if let playback {
                        PlayerView(player: playback.player)
                            .overlay(alignment: .topLeading) {
                                ReadoutView(values: ReadoutValues(result: output.result, pair: playback.pair), units: output.result.energyUnits)
                                    .padding(10)
                            }
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        FlowCharts(result: output.result, playback: playback)
                            .frame(height: 210)
                    }
                }
                .padding([.leading, .top, .bottom], 14)
                SummaryPanel(output: output)
                    .frame(width: 280)
                    .padding(14)
            }
            Divider()
            HStack {
                Button("Back to Setup") {
                    playback?.stop()
                    model.backToSetup()
                }
                Spacer()
                Text(String(format: "%d pairs analysed in %.1f s", output.result.pairs.count, output.result.elapsed))
                    .foregroundStyle(.secondary)
                Button("Export Movie…") { exportMovie() }
                    .buttonStyle(.borderedProminent)
            }
            .padding(12)
        }
        .task(id: output.movie) {
            playback?.stop()
            playback = PlaybackState(movie: output.movie, fps: output.fps, pairs: output.result.pairs.count)
        }
        .onDisappear { playback?.stop() }
    }

    private func exportMovie() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.quickTimeMovie]
        panel.nameFieldStringValue = output.result.info.url.deletingPathExtension().lastPathComponent + "_quiver.mov"
        guard panel.runModal() == .OK, let dest = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: dest.path) { try FileManager.default.removeItem(at: dest) }
            try FileManager.default.copyItem(at: output.movie, to: dest)
        } catch {
            model.error = "Couldn't save the movie: \(error.localizedDescription)"
        }
    }
}

struct PlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let v = AVPlayerView()
        v.player = player
        v.controlsStyle = .floating
        v.showsFrameSteppingButtons = true
        return v
    }

    func updateNSView(_ v: AVPlayerView, context: Context) {
        if v.player !== player { v.player = player }
    }
}

// MARK: - Readout

struct ReadoutView: View {
    let values: ReadoutValues
    let units: EnergyUnits

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(String(format: "t %.4f s   pair %d/%d", values.time, values.pair + 1, values.pairs))
                .foregroundStyle(.white)
            Text("ω mean \(signed(values.vorticityMean)) 1/s   |ω| \(ReadoutValues.format(Double(values.vorticityAbsMean))) 1/s")
                .foregroundStyle(.white)
            Text("KE \(ReadoutValues.format(values.keTotal.value)) \(values.keTotal.unit)   (\(Int((values.coverage * 100).rounded()))% measured)")
                .foregroundStyle(Color(red: 0.4, green: 0.9, blue: 0.75))
            Text("KE/mass \(ReadoutValues.format(values.keMean.value)) \(values.keMean.unit)")
                .foregroundStyle(Color(red: 0.4, green: 0.9, blue: 0.75))
        }
        .font(.system(.callout, design: .monospaced))
        .padding(8)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
        .allowsHitTesting(false)
    }

    private func signed(_ x: Float) -> String {
        x.isFinite ? (x >= 0 ? "+" : "") + ReadoutValues.format(Double(x)) : "–"
    }
}

// MARK: - Charts

struct FlowCharts: View {
    let result: RunResult
    let playback: PlaybackState

    private struct Point: Identifiable {
        let id: Int
        let t: Double
        let value: Double
        let series: String
    }

    private func points(_ stats: [FlowStats], _ series: String, _ f: (FlowStats) -> Double) -> [Point] {
        zip(result.pairs, stats).enumerated().compactMap { i, p in
            let v = f(p.1)
            return v.isFinite ? Point(id: i, t: p.0.time, value: v, series: series) : nil
        }
    }

    var body: some View {
        let smoothed = result.smoothedStats != nil
        let keUnit = result.energyUnits.keTotal(0).unit
        let w = points(result.stats, smoothed ? "smoothed" : "raw") { Double($0.vorticityMean) }
        let wRaw = smoothed ? points(result.rawStats, "raw") { Double($0.vorticityMean) } : []
        let ke = points(result.stats, smoothed ? "smoothed" : "raw") { result.energyUnits.keTotal($0.keSum).value }
        let keRaw = smoothed ? points(result.rawStats, "raw") { result.energyUnits.keTotal($0.keSum).value } : []
        let now = result.pairs[min(playback.pair, result.pairs.count - 1)].time
        VStack(spacing: 6) {
            chart(title: "Mean vorticity (1/s, + = counter-clockwise)", main: w, raw: wRaw, now: now, color: .primary, zeroLine: true)
            chart(title: "Total kinetic energy (\(keUnit), in-plane)", main: ke, raw: keRaw, now: now, color: .teal, zeroLine: false)
        }
    }

    private func chart(title: String, main: [Point], raw: [Point], now: Double, color: Color, zeroLine: Bool) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Chart {
                if zeroLine { RuleMark(y: .value("zero", 0)).foregroundStyle(.secondary.opacity(0.4)) }
                ForEach(raw) { p in
                    LineMark(x: .value("t", p.t), y: .value("value", p.value), series: .value("series", "raw"))
                        .foregroundStyle(.gray.opacity(0.45))
                        .lineStyle(StrokeStyle(lineWidth: 0.7))
                }
                ForEach(main) { p in
                    LineMark(x: .value("t", p.t), y: .value("value", p.value), series: .value("series", p.series))
                        .foregroundStyle(color)
                        .lineStyle(StrokeStyle(lineWidth: 1.1))
                }
                RuleMark(x: .value("now", now)).foregroundStyle(.orange)
            }
            .chartXAxisLabel("t (s)", alignment: .trailing)
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .gesture(DragGesture(minimumDistance: 0).onChanged { g in
                            guard let plot = proxy.plotFrame else { return }
                            let x = g.location.x - geo[plot].origin.x
                            if let t: Double = proxy.value(atX: x) { seek(to: t) }
                        })
                }
            }
        }
    }

    private func seek(to t: Double) {
        // Nearest pair by time.
        let i = result.pairs.indices.min { abs(result.pairs[$0].time - t) < abs(result.pairs[$1].time - t) } ?? 0
        playback.player.pause()
        playback.seek(toPair: i)
    }
}

// MARK: - Summary

struct SummaryPanel: View {
    let output: AppModel.RunOutput

    var body: some View {
        let s = output.summary, r = output.result
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                section("Vectors") {
                    row("Valid", pct(s.valid))
                    row("Rejected", "\(pct(s.rejected)) (\(pct(s.replaced)) replaced)")
                    row("No signal", pct(s.noSignal))
                }
                section("Displacement per pair") {
                    row("Median", String(format: "%.2f px", s.medianDisplacement))
                    row("95th / 99th pct", String(format: "%.2f / %.2f px", s.p95Displacement, s.p99Displacement))
                    row("Noise (one component)", String(format: "≈ %.2f px", s.noise))
                    if let hint = s.strideHint { Text(hint).font(.caption).foregroundStyle(.orange) }
                }
                section("Run") {
                    row("Capture rate", "\(r.capture.rate)")
                    row("Pair separation", String(format: "%.4g ms", r.dt * 1e3))
                    row("Grid", "\(r.grid.nx)×\(r.grid.ny), \(r.grid.step) px step")
                    if let sc = r.scale { row("Scale", String(format: "%.4f mm/px", sc.metresPerPixel * 1e3)) }
                    if let ts = r.request.temporalSmoothing {
                        row("Smoothing", "\(ts.kernel.rawValue) over \(ts.window) pairs")
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func pct(_ x: Double) -> String { String(format: "%.1f %%", x * 100) }

    private func section(_ title: String, @ViewBuilder _ content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.headline)
            content()
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit()
        }
        .font(.callout)
    }
}
