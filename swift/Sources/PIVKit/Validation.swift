import Foundation

/// Outlier detection and replacement. Port of `piv_core::validate`.
public struct ValidationConfig: Hashable, Sendable, Codable {
    /// Vectors whose primary/secondary peak ratio is below this are flagged `lowPPR`.
    public var pprThreshold: Float = 1.3
    /// Normalized median test threshold (Westerweel & Scarano 2005 recommend 2).
    public var medianThreshold: Float = 2
    /// Normalized median test noise level ε, in pixels.
    public var medianEps: Float = 0.1
    /// Replace rejected vectors with the median of valid neighbours.
    public var replace = true

    public init() {}
}

/// Limits that depend on the interrogation setup rather than on the user's tolerances.
public struct ValidationLimits: Hashable, Sendable {
    /// Largest displacement magnitude (px) the analysis can measure; larger is flagged
    /// `outOfRange`. Half the first-pass window: beyond that the correlation peak aliases.
    public var maxDisplacement: Float
    public init(maxDisplacement: Float) { self.maxDisplacement = maxDisplacement }
}

public struct ValidationSummary: Hashable, Sendable, Codable {
    public var noSignal = 0
    public var lowPPR = 0
    public var medianOutliers = 0
    public var outOfRange = 0
    public var replaced = 0
    public init() {}
}

/// Flags low-confidence vectors and outliers in place and (optionally) replaces them.
///
/// `noSignal` vectors are left as NaN: they are missing data, not errors.
@discardableResult
public func validate(
    _ field: inout VectorField, config cfg: ValidationConfig, limits: ValidationLimits
) -> ValidationSummary {
    let (nx, ny) = (field.grid.nx, field.grid.ny)
    var summary = ValidationSummary()

    for k in field.flags.indices {
        if field.flags[k].contains(.noSignal) {
            summary.noSignal += 1
            continue
        }
        if hypotf(field.dx[k], field.dy[k]) > limits.maxDisplacement {
            field.flags[k].insert(.outOfRange)
            summary.outOfRange += 1
        }
        if field.peakRatio[k] < cfg.pprThreshold {
            field.flags[k].insert(.lowPPR)
            summary.lowPPR += 1
        }
    }

    var nu: [Float] = []
    var nv: [Float] = []
    nu.reserveCapacity(24)
    nv.reserveCapacity(24)
    let usable = { (f: VectorFlags) in f.isValid }

    var snapshot = field.flags
    for iy in 0..<ny {
        for ix in 0..<nx where usable(snapshot[iy * nx + ix]) {
            collectNeighbours(field, snapshot, usable, ix, iy, 1, &nu, &nv)
            if nu.count < 2 { continue }
            let k = iy * nx + ix
            let ru = residual(field.dx[k], &nu, cfg.medianEps)
            let rv = residual(field.dy[k], &nv, cfg.medianEps)
            if (ru * ru + rv * rv).squareRoot() > cfg.medianThreshold {
                field.flags[k].insert(.medianOutlier)
                summary.medianOutliers += 1
            }
        }
    }

    if cfg.replace {
        snapshot = field.flags
        let rejected = { (f: VectorFlags) in !f.contains(.noSignal) && !f.isValid }
        for iy in 0..<ny {
            for ix in 0..<nx where rejected(snapshot[iy * nx + ix]) {
                let k = iy * nx + ix
                var replaced = false
                for radius in 1...2 {
                    collectNeighbours(field, snapshot, usable, ix, iy, radius, &nu, &nv)
                    if !nu.isEmpty {
                        field.dx[k] = median(&nu)
                        field.dy[k] = median(&nv)
                        field.flags[k].insert(.replaced)
                        summary.replaced += 1
                        replaced = true
                        break
                    }
                }
                if !replaced {
                    field.dx[k] = .nan
                    field.dy[k] = .nan
                }
            }
        }
    }
    return summary
}

private func collectNeighbours(
    _ field: VectorField, _ flags: [VectorFlags], _ usable: (VectorFlags) -> Bool,
    _ ix: Int, _ iy: Int, _ radius: Int, _ nu: inout [Float], _ nv: inout [Float]
) {
    let (nx, ny) = (field.grid.nx, field.grid.ny)
    nu.removeAll(keepingCapacity: true)
    nv.removeAll(keepingCapacity: true)
    for y in max(iy - radius, 0)...min(iy + radius, ny - 1) {
        for x in max(ix - radius, 0)...min(ix + radius, nx - 1) {
            let k = y * nx + x
            if (x, y) != (ix, iy), usable(flags[k]) {
                nu.append(field.dx[k])
                nv.append(field.dy[k])
            }
        }
    }
}

/// Normalized median residual of `value` against its neighbours (which get reordered).
private func residual(_ value: Float, _ neighbours: inout [Float], _ eps: Float) -> Float {
    let med = median(&neighbours)
    var dev = neighbours.map { abs($0 - med) }
    return abs(value - med) / (median(&dev) + eps)
}

/// Median of `values` (reordered in place); the mean of the two middle values for even counts.
/// NaN sorts last, matching Rust's `total_cmp` for positive NaN.
func median(_ values: inout [Float]) -> Float {
    values.sort { totalOrderLess($0, $1) }
    let n = values.count
    return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2])
}

@inline(__always) func totalOrderLess(_ a: Float, _ b: Float) -> Bool {
    a.isTotallyOrdered(belowOrEqualTo: b) && !b.isTotallyOrdered(belowOrEqualTo: a)
}
