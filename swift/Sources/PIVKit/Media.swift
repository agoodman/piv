import AVFoundation
import CoreVideo
import Foundation

/// Which image plane to analyse.
public enum Channel: String, Hashable, Sendable, Codable, CaseIterable {
    /// The Y plane exactly as stored: no range or colour conversion (`piv --channel yplane`).
    case luma
    /// Green after AVFoundation's conversion to RGB. Close to, but not the same as, the Rust
    /// tool's FFmpeg conversion.
    case green
}

/// Region of interest in full-frame pixel coordinates.
public struct ROI: Hashable, Sendable, Codable {
    public var x: Int, y: Int, width: Int, height: Int
    public init(x: Int, y: Int, width: Int, height: Int) {
        (self.x, self.y, self.width, self.height) = (x, y, width, height)
    }

    /// Parses `x,y,w,h`.
    public init(parsing s: String) throws(PIVError) {
        let v = s.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard v.count == 4, let x = v[0], let y = v[1], let w = v[2], let h = v[3], x >= 0, y >= 0, w > 0, h > 0 else {
            throw .config("invalid ROI \(s.debugDescription); expected x,y,w,h with w,h > 0")
        }
        self.init(x: x, y: y, width: w, height: h)
    }
}

/// What AVFoundation and the camera probes found out about a video file.
public struct MediaInfo: Sendable {
    public var url: URL
    public var fileSize: Int64
    public var codec: String
    public var width: Int
    public var height: Int
    public var bitDepth: Int
    /// Whether luma uses the full 0–255 range (otherwise 16–235 "video" range).
    public var fullRange: Bool
    public var frameCount: Int
    public var containerRate: FrameRate
    /// Presentation time of the first frame, s.
    public var startTime: Double
    public var cameraRate: CameraRate?

    public var playbackDuration: Double { containerRate.time(ofFrame: Double(frameCount)) }

    /// Probes `url` with AVFoundation plus the camera-metadata probes.
    public static func probe(_ url: URL) async throws -> MediaInfo {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw PIVError.io("\(url.lastPathComponent) has no video track")
        }
        let (size, minFrameDuration, formats, timeRange, nominal) = try await track.load(
            .naturalSize, .minFrameDuration, .formatDescriptions, .timeRange, .nominalFrameRate)
        guard let format = formats.first else { throw PIVError.io("\(url.lastPathComponent): no format description") }

        let containerRate = try Self.rate(minFrameDuration: minFrameDuration, nominal: nominal)
        let ext = CMFormatDescriptionGetExtensions(format) as? [CFString: Any] ?? [:]
        let bitDepth = (ext[kCMFormatDescriptionExtension_BitsPerComponent] as? Int) ?? 8
        let fullRange = (ext[kCMFormatDescriptionExtension_FullRangeVideo] as? Bool) ?? false
        let codec = fourCC(CMFormatDescriptionGetMediaSubType(format))

        // Count samples without decoding; fall back to duration × rate.
        var frameCount = Int((timeRange.duration.seconds * containerRate.fps).rounded())
        if try await track.load(.canProvideSampleCursors),
           let cursor = track.makeSampleCursorAtFirstSampleInDecodeOrder() {
            frameCount = Int(cursor.stepInDecodeOrder(byCount: Int64(Int32.max))) + 1
        }
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return MediaInfo(
            url: url, fileSize: (attrs?[.size] as? Int64) ?? 0, codec: codec,
            width: Int(size.width), height: Int(size.height), bitDepth: bitDepth, fullRange: fullRange,
            frameCount: frameCount, containerRate: containerRate, startTime: timeRange.start.seconds,
            cameraRate: try probeCameraRate(url)
        )
    }

    /// The exact container rate: from the frame duration when it is a clean rational, else from
    /// the nominal rate (snapping NTSC rates as `FrameRate(parsing:)` does).
    static func rate(minFrameDuration d: CMTime, nominal: Float) throws -> FrameRate {
        if d.isValid, d.value > 0, d.timescale > 0 {
            let r = try FrameRate(UInt64(d.timescale), UInt64(d.value))
            if abs(r.fps - Double(nominal)) < 0.01 * Double(nominal) || nominal == 0 { return r }
        }
        return try FrameRate(parsing: String(format: "%.3f", nominal))
    }
}

private func fourCC(_ code: FourCharCode) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xff) }
    return String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespaces)
}

/// A decoded frame: its index in the source and the pixel buffer AVFoundation produced.
public struct DecodedFrame: @unchecked Sendable {
    public let index: Int
    public let pixelBuffer: CVPixelBuffer
}

/// Streams frames `[start, end)` of a video in presentation order with AVFoundation, on its own
/// thread, at most `lookahead` frames ahead of the consumer.
///
/// Frames are indexed by presentation time, `round((pts − first pts) × container rate)`, which
/// matches the Rust tool's FFmpeg frame numbering for constant-rate files. For `.luma` the
/// output is the source's own Y′CbCr format, so the Y plane is untouched.
public final class VideoDecoder: @unchecked Sendable {
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    private let info: MediaInfo
    private let range: Range<Int>
    private let condition = NSCondition()
    private var queue: [DecodedFrame] = []
    private var finished = false
    private var failure: Error?
    private var cancelled = false
    private let lookahead: Int

    public init(info: MediaInfo, range: Range<Int>, channel: Channel, lookahead: Int = 8) async throws {
        let asset = AVURLAsset(url: info.url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw PIVError.io("\(info.url.lastPathComponent) has no video track")
        }
        self.info = info
        self.range = range
        self.lookahead = lookahead
        reader = try AVAssetReader(asset: asset)
        let pixelFormat: OSType = switch (channel, info.bitDepth > 8, info.fullRange) {
        case (.green, _, _): kCVPixelFormatType_32BGRA
        case (.luma, false, false): kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        case (.luma, false, true): kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        case (.luma, true, false): kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        case (.luma, true, true): kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        }
        output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        // Start exactly at the first frame's timestamp, as an exact rational. AVAssetReader
        // also emits a sample that merely overlaps the start of the range, with its timestamp
        // clipped to the range start, so starting even half a frame early labels frame k − 1's
        // picture as frame k.
        let rate = info.containerRate
        let first = CMTimeAdd(
            CMTime(seconds: info.startTime, preferredTimescale: 600_000),
            CMTime(value: Int64(range.lowerBound) * Int64(rate.den), timescale: CMTimeScale(rate.num)))
        reader.timeRange = CMTimeRange(start: first, duration: .positiveInfinity)
        guard reader.startReading() else {
            throw PIVError.io("cannot read \(info.url.lastPathComponent): \(reader.error?.localizedDescription ?? "unknown error")")
        }
        // The thread keeps the decoder alive until it finishes or is cancelled.
        let thread = Thread { self.run() }
        thread.name = "PIVKit decoder"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    deinit { reader.cancelReading() }

    private func run() {
        let rate = info.containerRate
        var lastIndex = Int.min
        while true {
            condition.lock()
            while queue.count >= lookahead && !cancelled { condition.wait() }
            let stop = cancelled
            condition.unlock()
            if stop { break }

            guard let sample = output.copyNextSampleBuffer() else {
                finish(reader.status == .failed ? PIVError.io("decoding failed: \(reader.error?.localizedDescription ?? "unknown")") : nil)
                return
            }
            guard let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds - info.startTime
            let index = Int((pts * Double(rate.num) / Double(rate.den)).rounded())
            // Never repeat an index (a clipped overlapping sample would).
            if index < range.lowerBound || index <= lastIndex { continue }
            lastIndex = index
            if index >= range.upperBound {
                finish(nil)
                return
            }
            condition.lock()
            queue.append(DecodedFrame(index: index, pixelBuffer: pb))
            condition.broadcast()
            condition.unlock()
        }
        finish(nil)
    }

    private func finish(_ error: Error?) {
        condition.lock()
        finished = true
        failure = error
        condition.broadcast()
        condition.unlock()
    }

    /// The next frame, or `nil` at the end of the range. Blocks until one is decoded.
    public func next() throws -> DecodedFrame? {
        condition.lock()
        defer { condition.unlock() }
        while queue.isEmpty && !finished { condition.wait() }
        if !queue.isEmpty {
            let f = queue.removeFirst()
            condition.broadcast()
            return f
        }
        if let failure { throw failure }
        return nil
    }

    /// ``next()`` without blocking a Swift concurrency thread.
    public func nextFrame() async throws -> DecodedFrame? {
        try await withCheckedThrowingContinuation { c in
            DispatchQueue.global(qos: .userInitiated).async {
                c.resume(with: Result { try self.next() })
            }
        }
    }

    /// Stops decoding (the thread exits at its next frame). Call this when abandoning a
    /// decoder before the end of its range.
    public func cancel() {
        condition.lock()
        cancelled = true
        condition.broadcast()
        condition.unlock()
        reader.cancelReading()
    }

    /// Decodes the single frame at `index`.
    public static func frame(_ index: Int, info: MediaInfo, channel: Channel) async throws -> DecodedFrame {
        let d = try await VideoDecoder(info: info, range: index..<(index + 1), channel: channel, lookahead: 1)
        defer { d.cancel() }
        guard let f = try await d.nextFrame(), f.index == index else {
            throw PIVError.io("frame \(index) not found in \(info.url.lastPathComponent)")
        }
        return f
    }
}
