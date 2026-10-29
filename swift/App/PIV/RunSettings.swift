import Foundation
import PIVKit

/// Everything the setup screen edits. Remembered per clip, and turned into an
/// `AnalysisRequest` when a run starts.
struct RunSettings: Codable, Equatable {
    enum ScaleMode: String, Codable, CaseIterable, Identifiable {
        case none, pxPerMM, fov
        var id: Self { self }
        var label: String {
            switch self {
            case .none: "None (pixels)"
            case .pxPerMM: "Measured px/mm"
            case .fov: "Field of view + distance"
            }
        }
    }

    enum BackgroundChoice: String, Codable, CaseIterable, Identifiable {
        case min, mean, none
        var id: Self { self }
    }

    // Timing
    var captureFPS = ""
    var startFrame = 0
    var endFrame = 0
    var stride = 1
    var pairStep = 1

    // Image
    var roi: ROI?
    var channel = Channel.luma

    // Scale
    var scaleMode = ScaleMode.none
    var pxPerMM = 4.0
    var fovDegrees = 84.0
    var fovAxis = FOVAxis.diagonal
    var distanceMM = 300.0
    var sheetThicknessMM: Double?
    var density = airDensity

    // Smoothing
    var smoothing = false
    var smoothingWindow = 9
    var smoothingKernel = TemporalSmoothing.Kernel.median

    // Advanced
    var windows = "96, 64"
    var overlap: Float = 0.5
    var finalIterations = 2
    var background = BackgroundChoice.min
    var backgroundFrames = 40
    var smoothSigma: Float = 1
    var normalizeRadius = 4
    var minMean = AnalysisRequest.defaultMask.minMean
    var minStd = AnalysisRequest.defaultMask.minStd
    var pprThreshold = ValidationConfig().pprThreshold
    var medianThreshold = ValidationConfig().medianThreshold
    var medianEps = ValidationConfig().medianEps
    var replace = true

    // Output
    var videoFPS = 30.0

    /// Defaults for a newly opened clip: the whole clip, capture rate from metadata.
    init(for info: MediaInfo) {
        let rate = info.cameraRate?.rate ?? info.containerRate
        captureFPS = rate.den == 1 ? "\(rate.num)" : "\(rate.num)/\(rate.den)"
        endFrame = info.frameCount
    }

    var captureRate: FrameRate? { try? FrameRate(parsing: captureFPS) }
    var windowSizes: [Int] {
        windows.split(whereSeparator: { $0 == "," || $0 == " " }).compactMap { Int($0) }
    }
    var pairCount: Int { max(0, (endFrame - stride - startFrame + pairStep - 1) / pairStep) }

    /// The analysis request, or a message saying what's wrong.
    func request(info: MediaInfo) -> Result<AnalysisRequest, SettingsError> {
        guard let rate = captureRate else { return .failure(.init("Capture frame rate \(captureFPS.debugDescription) isn't a rate (try 1000, 239.76 or 240000/1001).")) }
        guard stride >= 1, pairStep >= 1 else { return .failure(.init("Stride and pair step must be at least 1.")) }
        guard startFrame >= 0, endFrame <= info.frameCount, startFrame + stride < endFrame else {
            return .failure(.init("The frame range [\(startFrame), \(endFrame)) is too short for stride \(stride)."))
        }
        if let r = roi, r.x + r.width > info.width || r.y + r.height > info.height {
            return .failure(.init("The region of interest extends past the \(info.width)×\(info.height) frame."))
        }
        var req = AnalysisRequest(source: info.url)
        req.captureRate = rate
        req.frameRange = startFrame..<endFrame
        req.stride = stride
        req.pairStep = pairStep
        req.roi = roi
        req.channel = channel
        req.background = background == .none ? nil : BackgroundSpec(mode: background == .min ? .min : .mean, frames: backgroundFrames)
        req.preprocess.smoothSigma = smoothSigma
        req.preprocess.normalizeRadius = normalizeRadius
        req.piv.windows = windowSizes
        req.piv.overlap = overlap
        req.piv.finalIterations = finalIterations
        req.piv.mask = MaskConfig(minMean: minMean, minStd: minStd)
        req.piv.validation.pprThreshold = pprThreshold
        req.piv.validation.medianThreshold = medianThreshold
        req.piv.validation.medianEps = medianEps
        req.piv.validation.replace = replace
        switch scaleMode {
        case .none: req.scale = ScaleSpec()
        case .pxPerMM: req.scale = ScaleSpec(pxPerMM: pxPerMM)
        case .fov: req.scale = ScaleSpec(fovDegrees: fovDegrees, fovAxis: fovAxis, sheetDistanceMetres: distanceMM / 1000)
        }
        req.sheetThicknessMM = sheetThicknessMM
        req.density = density
        if smoothing { req.temporalSmoothing = TemporalSmoothing(window: smoothingWindow, kernel: smoothingKernel) }
        do {
            try req.piv.validate()
            try req.temporalSmoothing?.validate()
        } catch {
            return .failure(.init(error.localizedDescription))
        }
        return .success(req)
    }

    /// The resolved scale for display (mm/px), if the scale settings give one.
    func scale(info: MediaInfo) -> Scale? {
        switch scaleMode {
        case .none: nil
        case .pxPerMM: ScaleSpec(pxPerMM: pxPerMM).resolve(width: info.width, height: info.height)
        case .fov: ScaleSpec(fovDegrees: fovDegrees, fovAxis: fovAxis, sheetDistanceMetres: distanceMM / 1000)
            .resolve(width: info.width, height: info.height)
        }
    }

    // MARK: Remembering settings per clip

    private static func key(_ info: MediaInfo) -> String { "settings:\(info.url.path):\(info.fileSize)" }

    static func saved(for info: MediaInfo, in defaults: UserDefaults = .standard) -> RunSettings? {
        defaults.data(forKey: key(info)).flatMap { try? JSONDecoder().decode(RunSettings.self, from: $0) }
    }

    func save(for info: MediaInfo, in defaults: UserDefaults = .standard) {
        if let d = try? JSONEncoder().encode(self) { defaults.set(d, forKey: Self.key(info)) }
    }
}

struct SettingsError: Error, Equatable {
    let message: String
    init(_ message: String) { self.message = message }
}
