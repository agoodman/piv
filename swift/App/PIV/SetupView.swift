import AVFoundation
import PIVKit
import SwiftUI

struct SetupView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let info = model.info, model.settings != nil {
            HSplitView {
                FramePreview(info: info)
                    .frame(minWidth: 480)
                SettingsForm(info: info)
                    .frame(minWidth: 380, idealWidth: 420, maxWidth: 520)
            }
        }
    }
}

// MARK: - Preview with scrubber and ROI

struct FramePreview: View {
    @Environment(AppModel.self) private var model
    let info: MediaInfo
    @State private var frame = 0
    @State private var loader: FrameLoader?
    @State private var dragStart: CGPoint?
    @State private var dragRect: CGRect?

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 10) {
            ClipHeader(info: info)
            GeometryReader { geo in
                let fit = fitted(in: geo.size)
                ZStack(alignment: .topLeading) {
                    Color.black
                    if let image = loader?.image {
                        Image(decorative: image, scale: 1)
                            .resizable()
                            .frame(width: fit.width, height: fit.height)
                            .offset(x: fit.minX, y: fit.minY)
                    }
                    if let r = dragRect ?? model.settings?.roi.map(rect(of:)) {
                        let v = toView(r, fit)
                        Rectangle()
                            .strokeBorder(Color.yellow, style: StrokeStyle(lineWidth: 1.5, dash: [6, 3]))
                            .background(Color.yellow.opacity(0.08))
                            .frame(width: v.width, height: v.height)
                            .offset(x: v.minX, y: v.minY)
                    }
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 4)
                        .onChanged { g in
                            let a = toImage(g.startLocation, fit), b = toImage(g.location, fit)
                            dragRect = CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
                        }
                        .onEnded { _ in
                            if let r = dragRect, r.width >= 32, r.height >= 32 {
                                model.settings?.roi = ROI(x: Int(r.minX), y: Int(r.minY), width: Int(r.width), height: Int(r.height))
                            }
                            dragRect = nil
                        }
                )
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            HStack {
                Text("Frame")
                Slider(value: Binding(get: { Double(frame) }, set: { frame = Int($0) }),
                       in: 0...Double(max(info.frameCount - 1, 1))) { editing in
                    // Settle on the exact frame when the drag ends.
                    if !editing { loader?.request(frame, exact: true) }
                }
                Text("\(frame)")
                    .monospacedDigit()
                    .frame(width: 56, alignment: .trailing)
            }
            HStack {
                Button("Set Start") { model.settings?.startFrame = frame }
                Button("Set End") { model.settings?.endFrame = frame + 1 }
                Spacer()
                Text("Drag on the frame to set the region of interest.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if model.settings?.roi != nil {
                    Button("Clear Region") { model.settings?.roi = nil }
                }
            }
        }
        .padding()
        .onChange(of: frame) { _, f in loader?.request(f, exact: false) }
        .task(id: info.url) {
            let l = FrameLoader(info: info)
            loader = l
            l.request(frame, exact: true)
        }
    }

    private func rect(of r: ROI) -> CGRect { CGRect(x: r.x, y: r.y, width: r.width, height: r.height) }

    /// Where the frame sits in the view (aspect fit).
    private func fitted(in size: CGSize) -> CGRect {
        let (w, h) = (CGFloat(info.width), CGFloat(info.height))
        let s = min(size.width / w, size.height / h)
        return CGRect(x: (size.width - w * s) / 2, y: (size.height - h * s) / 2, width: w * s, height: h * s)
    }

    private func toImage(_ p: CGPoint, _ fit: CGRect) -> CGPoint {
        let s = fit.width / CGFloat(info.width)
        return CGPoint(x: min(max((p.x - fit.minX) / s, 0), CGFloat(info.width)),
                       y: min(max((p.y - fit.minY) / s, 0), CGFloat(info.height)))
    }

    private func toView(_ r: CGRect, _ fit: CGRect) -> CGRect {
        let s = fit.width / CGFloat(info.width)
        return CGRect(x: fit.minX + r.minX * s, y: fit.minY + r.minY * s, width: r.width * s, height: r.height * s)
    }
}

struct ClipHeader: View {
    let info: MediaInfo

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(info.url.lastPathComponent).font(.headline)
                Text(verbatim: "\(info.width)×\(info.height) · \(info.bitDepth)-bit \(info.codec) · \(info.frameCount) frames at \(info.containerRate) (container)\(info.cameraRate?.camera.map { " · \($0)" } ?? "")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }
}

// MARK: - Settings

struct SettingsForm: View {
    @Environment(AppModel.self) private var model
    let info: MediaInfo
    @State private var showAdvanced = false

    var body: some View {
        @Bindable var model = model
        if let settings = Binding($model.settings) {
            VStack(spacing: 0) {
                Form {
                    timing(settings)
                    region(settings)
                    scale(settings)
                    smoothing(settings)
                    Section {
                        DisclosureGroup("Advanced", isExpanded: $showAdvanced) { advanced(settings) }
                    }
                }
                .formStyle(.grouped)
                Divider()
                HStack {
                    Button("Reset to Defaults") { model.resetSettings() }
                    Spacer()
                    Text("\(settings.wrappedValue.pairCount) pairs")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    Button("Analyse") { model.run() }
                        .keyboardShortcut(.return, modifiers: .command)
                        .buttonStyle(.borderedProminent)
                }
                .padding(12)
            }
        }
    }

    @ViewBuilder private func timing(_ s: Binding<RunSettings>) -> some View {
        Section("Timing") {
            TextField("Capture frame rate (fps)", text: s.captureFPS)
            if let note = model.captureRateNote {
                Text(note).font(.caption).foregroundStyle(s.wrappedValue.captureRate == nil ? .red : .secondary)
            }
            Stepper(value: s.stride, in: 1...1000) { LabeledContent("Stride (frames per pair)", value: "\(s.wrappedValue.stride)") }
            if let rate = s.wrappedValue.captureRate {
                LabeledContent("Time between frames of a pair", value: String(format: "%.4g ms", rate.dt(stride: s.wrappedValue.stride) * 1e3))
            }
            Stepper(value: s.pairStep, in: 1...1000) { LabeledContent("Pair step", value: "\(s.wrappedValue.pairStep)") }
            LabeledContent("Frames") {
                HStack {
                    TextField("Start", value: s.startFrame, format: .number).frame(width: 70)
                    Text("to")
                    TextField("End", value: s.endFrame, format: .number).frame(width: 70)
                    Text("of \(info.frameCount)").foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder private func region(_ s: Binding<RunSettings>) -> some View {
        Section("Region") {
            if let r = s.wrappedValue.roi {
                LabeledContent("Region of interest", value: "\(r.width)×\(r.height) at (\(r.x), \(r.y))")
            } else {
                LabeledContent("Region of interest", value: "Whole frame")
            }
            Picker("Channel", selection: s.channel) {
                Text("Luma (Y plane)").tag(Channel.luma)
                Text("Green").tag(Channel.green)
            }
        }
    }

    @ViewBuilder private func scale(_ s: Binding<RunSettings>) -> some View {
        Section("Scale") {
            Picker("Scale from", selection: s.scaleMode) {
                ForEach(RunSettings.ScaleMode.allCases) { Text($0.label).tag($0) }
            }
            switch s.wrappedValue.scaleMode {
            case .none:
                EmptyView()
            case .pxPerMM:
                TextField("Pixels per mm", value: s.pxPerMM, format: .number)
            case .fov:
                TextField("Field of view (°)", value: s.fovDegrees, format: .number)
                Picker("Measured across", selection: s.fovAxis) {
                    Text("Diagonal").tag(FOVAxis.diagonal)
                    Text("Horizontal").tag(FOVAxis.horizontal)
                    Text("Vertical").tag(FOVAxis.vertical)
                }
                TextField("Camera to sheet (mm)", value: s.distanceMM, format: .number)
            }
            if let sc = s.wrappedValue.scale(info: info) {
                LabeledContent("Resolved", value: String(format: "%.4f mm/px", sc.metresPerPixel * 1e3))
                TextField("Sheet thickness (mm, 0 = unknown)", value: Binding(
                    get: { s.wrappedValue.sheetThicknessMM ?? 0 },
                    set: { s.wrappedValue.sheetThicknessMM = $0 > 0 ? $0 : nil }), format: .number)
                TextField("Fluid density (kg/m³)", value: s.density, format: .number)
            }
        }
    }

    @ViewBuilder private func smoothing(_ s: Binding<RunSettings>) -> some View {
        Section("Smoothing") {
            Toggle("Temporal smoothing", isOn: s.smoothing)
            if s.wrappedValue.smoothing {
                Stepper(value: s.smoothingWindow, in: 3...99, step: 2) {
                    LabeledContent("Window", value: windowLabel(s.wrappedValue))
                }
                Picker("Kernel", selection: s.smoothingKernel) {
                    Text("Median").tag(TemporalSmoothing.Kernel.median)
                    Text("Mean").tag(TemporalSmoothing.Kernel.mean)
                }
            }
        }
    }

    private func windowLabel(_ s: RunSettings) -> String {
        guard let rate = s.captureRate else { return "\(s.smoothingWindow) pairs" }
        let ms = rate.dt(stride: s.pairStep) * Double(s.smoothingWindow) * 1e3
        return "\(s.smoothingWindow) pairs (≈\(String(format: "%.3g", ms)) ms)"
    }

    @ViewBuilder private func advanced(_ s: Binding<RunSettings>) -> some View {
        TextField("Window sizes, largest first (px)", text: s.windows)
        TextField("Window overlap", value: s.overlap, format: .number)
        Stepper(value: s.finalIterations, in: 1...5) { LabeledContent("Final-size passes", value: "\(s.wrappedValue.finalIterations)") }
        Picker("Background", selection: s.background) {
            Text("Minimum").tag(RunSettings.BackgroundChoice.min)
            Text("Mean").tag(RunSettings.BackgroundChoice.mean)
            Text("None").tag(RunSettings.BackgroundChoice.none)
        }
        if s.wrappedValue.background != .none {
            TextField("Background frames", value: s.backgroundFrames, format: .number)
        }
        TextField("Gaussian smoothing σ (px)", value: s.smoothSigma, format: .number)
        TextField("Contrast normalization radius (px)", value: s.normalizeRadius, format: .number)
        TextField("Mask: minimum mean", value: s.minMean, format: .number)
        TextField("Mask: minimum std", value: s.minStd, format: .number)
        TextField("Minimum peak ratio", value: s.pprThreshold, format: .number)
        TextField("Median test threshold", value: s.medianThreshold, format: .number)
        TextField("Median test ε (px)", value: s.medianEps, format: .number)
        Toggle("Replace rejected vectors", isOn: s.replace)
        TextField("Movie frame rate", value: s.videoFPS, format: .number)
    }
}
