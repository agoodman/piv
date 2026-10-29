import Foundation

/// An exact frame rate, stored as a reduced rational so that rates like 240000/1001 survive
/// round trips without drift. Port of `piv_core::FrameRate`.
public struct FrameRate: Hashable, Sendable, Codable, CustomStringConvertible {
    public let num: UInt64
    public let den: UInt64

    public init(_ num: UInt64, _ den: UInt64 = 1) throws(PIVError) {
        guard num > 0, den > 0 else { throw .frameRate("\(num)/\(den)") }
        let g = gcd(num, den)
        self.num = num / g
        self.den = den / g
    }

    public var fps: Double { Double(num) / Double(den) }

    /// Seconds between frame `i` and frame `i + stride`.
    public func dt(stride: Int) -> Double { Double(stride) * Double(den) / Double(num) }

    /// Time of (possibly fractional) frame `index`, in seconds from frame 0.
    public func time(ofFrame index: Double) -> Double { index * Double(den) / Double(num) }

    /// Accepts `240`, `239.76`, `240000/1001`, and camera-style strings such as `1000p`,
    /// `59.94i` or `240fps`. Decimal values within 0.006 of an NTSC rate (`n·1000/1001`) are
    /// snapped to that exact rational.
    public init(parsing s: String) throws(PIVError) {
        let err = PIVError.frameRate(s)
        var t = s.trimmingCharacters(in: .whitespaces).lowercased()
        for suffix in ["fps", "p", "i"] where t.hasSuffix(suffix) {
            t.removeLast(suffix.count)
            break
        }
        t = t.trimmingCharacters(in: .whitespaces)

        if let slash = t.firstIndex(of: "/") {
            guard let n = UInt64(t[..<slash].trimmingCharacters(in: .whitespaces)),
                  let d = UInt64(t[t.index(after: slash)...].trimmingCharacters(in: .whitespaces))
            else { throw err }
            do { try self.init(n, d) } catch { throw err }
            return
        }
        if let dot = t.firstIndex(of: ".") {
            let int = t[..<dot]
            let frac = t[t.index(after: dot)...]
            guard let f = Double(t), f.isFinite, f > 0, !frac.isEmpty,
                  frac.allSatisfy({ $0.isASCII && $0.isNumber }),
                  int.allSatisfy({ $0.isASCII && $0.isNumber })
            else { throw err }
            let n = (f * 1001 / 1000).rounded()
            if n >= 1, abs(n * 1000 / 1001 - f) < 0.006 {
                try self.init(UInt64(n) * 1000, 1001)
                return
            }
            guard frac.count <= 19, let num = UInt64(int + frac) else { throw err }
            var den: UInt64 = 1
            for _ in 0..<frac.count { den *= 10 }
            do { try self.init(num, den) } catch { throw err }
            return
        }
        guard let n = UInt64(t) else { throw err }
        do { try self.init(n, 1) } catch { throw err }
    }

    public var description: String {
        den == 1 ? "\(num) fps" : "\(num)/\(den) (\(String(format: "%.3f", fps))) fps"
    }
}

private func gcd(_ a: UInt64, _ b: UInt64) -> UInt64 {
    var (a, b) = (a, b)
    while b != 0 { (a, b) = (b, a % b) }
    return a
}
