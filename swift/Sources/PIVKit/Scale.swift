import Foundation

/// Approximate physical scale (metres per pixel) from user-entered optics. Port of
/// `piv_io::scale`.
///
/// This is not a calibration: it assumes a distortion-free pinhole camera looking straight at
/// the light sheet. A measured px/mm from a ruler or target in the sheet plane is always
/// preferable.
public enum FOVAxis: String, Hashable, Sendable, Codable, CaseIterable {
    case diagonal, horizontal, vertical
}

/// What the user told us about the optics.
public struct ScaleSpec: Hashable, Sendable, Codable {
    public var pxPerMM: Double?
    public var fovDegrees: Double?
    public var fovAxis: FOVAxis?
    public var sheetDistanceMetres: Double?

    public init(
        pxPerMM: Double? = nil, fovDegrees: Double? = nil, fovAxis: FOVAxis? = nil,
        sheetDistanceMetres: Double? = nil
    ) {
        self.pxPerMM = pxPerMM
        self.fovDegrees = fovDegrees
        self.fovAxis = fovAxis
        self.sheetDistanceMetres = sheetDistanceMetres
    }

    /// Resolves a scale for a full frame of `width × height` pixels. Returns `nil` when the
    /// inputs are insufficient (e.g. a field of view without a distance).
    public func resolve(width: Int, height: Int) -> Scale? {
        if let s = pxPerMM {
            return Scale(metresPerPixel: 1e-3 / s, method: "--px-per-mm \(fmt(s))")
        }
        guard let fov = fovDegrees, let d = sheetDistanceMetres else { return nil }
        let axis = fovAxis ?? .diagonal
        let (w, h) = (Double(width), Double(height))
        let spanPx = switch axis {
        case .diagonal: hypot(w, h)
        case .horizontal: w
        case .vertical: h
        }
        let spanM = 2 * d * tan(fov * .pi / 180 / 2)
        return Scale(
            metresPerPixel: spanM / spanPx,
            method: "pinhole estimate: \(fmt(fov))° \(axis.rawValue) fov at \(fmt(d)) m over \(String(format: "%.1f", spanPx)) px"
        )
    }
}

public struct Scale: Hashable, Sendable, Codable {
    public var metresPerPixel: Double
    public var method: String
}

/// Shortest decimal form, like Rust's `{}` for f64 (`0.3`, `84`).
private func fmt(_ x: Double) -> String {
    x == x.rounded() && abs(x) < 1e15 ? String(Int64(x)) : "\(x)"
}
