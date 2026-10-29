import Foundation

/// Capture-rate discovery. Port of the camera probes and resolution order in
/// `piv_io::metadata`.
///
/// High-speed cameras often record at one rate and write the file at a slower playback rate
/// (C0141.MP4: captured at 1000 fps, stored as 50p). Velocities must use the capture rate, so
/// it is resolved from, in order: a user override, camera-specific metadata, the container.

/// A capture rate read from camera-specific metadata.
public struct CameraRate: Hashable, Sendable, Codable {
    public var rate: FrameRate
    /// Which probe found it, e.g. "Sony NonRealTimeMeta".
    public var source: String
    /// The raw value as written by the camera, e.g. "1000p".
    public var raw: String
    /// Camera make and model, if the metadata names it.
    public var camera: String?

    public init(rate: FrameRate, source: String, raw: String, camera: String? = nil) {
        self.rate = rate
        self.source = source
        self.raw = raw
        self.camera = camera
    }
}

/// The rate used for timing, and where it came from.
public struct CaptureRate: Hashable, Sendable {
    public enum Source: String, Hashable, Sendable { case override, cameraMetadata, container }
    public var rate: FrameRate
    public var source: Source
    public var detail: String
    public var warning: String?

    /// Picks the capture rate: `override`, then camera metadata, then the container rate (with
    /// a warning, since slow-motion footage would make that wrong).
    public static func resolve(
        override: FrameRate?, camera: CameraRate?, container: FrameRate
    ) -> CaptureRate {
        if let rate = override {
            let warning = camera.flatMap { c in
                c.rate == rate ? nil : "capture rate \(rate) overrides \(c.rate) from \(c.source) metadata"
            }
            return CaptureRate(rate: rate, source: .override, detail: "user override", warning: warning)
        }
        if let c = camera {
            return CaptureRate(rate: c.rate, source: .cameraMetadata, detail: "\(c.source) (\(c.raw))")
        }
        return CaptureRate(
            rate: container, source: .container, detail: "container frame rate",
            warning: "no capture-rate metadata found; assuming the container rate of \(container) "
                + "is real time — set the capture rate if this footage is slowed down"
        )
    }
}

/// Camera-specific capture-rate metadata. Implementations return `nil` when the file is not
/// from their camera family.
public protocol CaptureRateProbe: Sendable {
    var name: String { get }
    func probe(_ url: URL) throws -> CameraRate?
}

/// All known probes, tried in order.
public let cameraProbes: [any CaptureRateProbe] = [SonyNRTProbe()]

/// The first camera rate any probe finds.
public func probeCameraRate(_ url: URL) throws -> CameraRate? {
    for p in cameraProbes {
        if let r = try p.probe(url) { return r }
    }
    return nil
}

/// Sony XAVC S files carry a `NonRealTimeMeta` XML document (in a top-level `meta` box)
/// whose `VideoFrame@captureFps` is the true sensor rate in S&Q / HFR modes.
public struct SonyNRTProbe: CaptureRateProbe {
    public init() {}
    public var name: String { "Sony NonRealTimeMeta" }

    public func probe(_ url: URL) throws -> CameraRate? {
        guard let xml = try findMP4Text(url, start: "<NonRealTimeMeta", end: "</NonRealTimeMeta>"),
              let raw = xmlAttr(xml, tag: "VideoFrame", attr: "captureFps")
        else { return nil }
        let rate: FrameRate
        do { rate = try FrameRate(parsing: raw) } catch {
            throw PIVError.io("captureFps \(raw.debugDescription): \(error.localizedDescription)")
        }
        let camera: String? = switch (
            xmlAttr(xml, tag: "Device", attr: "manufacturer"), xmlAttr(xml, tag: "Device", attr: "modelName")
        ) {
        case let (m?, n?): "\(m) \(n)"
        case let (m, n): m ?? n
        }
        return CameraRate(rate: rate, source: name, raw: raw, camera: camera)
    }
}

/// Searches the top-level boxes of an ISO-BMFF (MP4/MOV) file, other than the media data,
/// for a text span starting with `start` and ending with `end`.
func findMP4Text(_ url: URL, start: String, end: String) throws -> String? {
    let maxBox: UInt64 = 16 << 20
    let f: FileHandle
    do { f = try FileHandle(forReadingFrom: url) } catch {
        throw PIVError.io("opening \(url.path): \(error.localizedDescription)")
    }
    defer { try? f.close() }
    let len = try f.seekToEnd()
    var off: UInt64 = 0
    while off + 8 <= len {
        try f.seek(toOffset: off)
        let hdr = [UInt8](try f.read(upToCount: 16) ?? Data())
        if hdr.count < 8 { break }
        let size32 = hdr[0..<4].reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        let kind = String(decoding: hdr[4..<8], as: UTF8.self)
        let size: UInt64
        switch size32 {
        case 0: size = len - off
        case 1 where hdr.count >= 16: size = hdr[8..<16].reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        case 1: return nil
        default: size = size32
        }
        if size < 8 { break } // corrupt; stop rather than loop forever
        if !["mdat", "free", "skip"].contains(kind), size <= maxBox {
            try f.seek(toOffset: off)
            guard let buf = try f.read(upToCount: Int(size)), buf.count == Int(size) else {
                throw PIVError.io("\(url.path): truncated \(kind) box")
            }
            if let text = findSpan(buf, start: Data(start.utf8), end: Data(end.utf8)) {
                return String(decoding: text, as: UTF8.self)
            }
        }
        off += size
    }
    return nil
}

private func findSpan(_ buf: Data, start: Data, end: Data) -> Data? {
    guard let s = buf.range(of: start),
          let e = buf.range(of: end, in: s.lowerBound..<buf.endIndex)
    else { return nil }
    return buf[s.lowerBound..<e.upperBound]
}

/// Value of attribute `attr` on the first `<tag ...>` element. Enough for the flat,
/// machine-written XML cameras emit; not a general XML parser.
func xmlAttr(_ xml: String, tag: String, attr: String) -> String? {
    let open = "<\(tag)"
    var rest = Substring(xml)
    var elem: Substring
    while true {
        guard let i = rest.range(of: open) else { return nil }
        let after = rest[i.upperBound...]
        if let c = after.first, c.isWhitespace || c == ">" || c == "/" {
            guard let close = after.firstIndex(of: ">") else { return nil }
            elem = after[..<close]
            break
        }
        rest = after
    }
    let key = "\(attr)=\""
    var search = elem
    while true {
        guard let i = search.range(of: key) else { return nil }
        if i.lowerBound == search.startIndex || search[search.index(before: i.lowerBound)].isWhitespace {
            let v = search[i.upperBound...]
            guard let q = v.firstIndex(of: "\"") else { return nil }
            return String(v[..<q])
        }
        search = search[i.upperBound...]
    }
}
