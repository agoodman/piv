/// Moving mean or median over consecutive pairs at each grid point. Port of
/// `piv_core::smooth::temporal`.
public struct TemporalSmoothing: Hashable, Sendable, Codable {
    public enum Kernel: String, Hashable, Sendable, Codable, CaseIterable {
        case mean, median
    }

    /// Number of consecutive pairs (odd, centred).
    public var window: Int
    public var kernel: Kernel
    /// Raw vectors needed in the window; `nil` = ⌈window / 3⌉, as in `piv smooth`.
    public var minSupport: Int?

    public init(window: Int = 9, kernel: Kernel = .median, minSupport: Int? = nil) {
        self.window = window
        self.kernel = kernel
        self.minSupport = minSupport
    }

    /// The effective minimum support: `minSupport`, or ⌈window / 3⌉, at most `window`.
    public var resolvedMinSupport: Int { min(minSupport ?? (window + 2) / 3, window) }

    public func validate() throws(PIVError) {
        guard window >= 1, window % 2 == 1 else {
            throw .config("temporal window \(window) must be odd and ≥ 1")
        }
        if let m = minSupport, m > window {
            throw .config("min support \(m) exceeds the \(window)-pair temporal window")
        }
    }

    /// Smooths a run the way `piv smooth --temporal` does: accepted, finite raw vectors
    /// contribute, outputs are produced wherever the window had seeding in that pair, and
    /// smoothed vectors carry no flags (or `noSignal` where nothing was output). Peak ratio has
    /// no meaning for a smoothed vector and is set to 0.
    public func apply(to fields: [VectorField]) -> (fields: [VectorField], support: [[UInt16]]) {
        guard let grid = fields.first?.grid else { return ([], []) }
        let usable = fields.map { f in f.flags.indices.map { k in f.flags[k].isValid && f.dx[k].isFinite && f.dy[k].isFinite } }
        let gate = fields.map { $0.flags.map { !$0.contains(.noSignal) } }
        let s = temporalSmooth(
            dx: fields.map(\.dx), dy: fields.map(\.dy), usable: usable, gate: gate,
            kernel: kernel, window: window, minSupport: resolvedMinSupport
        )
        let out = s.dx.indices.map { t in
            VectorField(
                grid: grid, dx: s.dx[t], dy: s.dy[t],
                peakRatio: Array(repeating: 0, count: grid.count),
                flags: s.dx[t].map { $0.isFinite ? [] : .noSignal }
            )
        }
        return (out, s.support)
    }
}

/// A smoothed series, `[pair][ny][nx]` flattened per pair.
public struct SmoothedSeries: Sendable, Equatable {
    public var dx: [[Float]]
    public var dy: [[Float]]
    /// Number of usable raw vectors in the window (may be below `minSupport` where there is no
    /// output).
    public var support: [[UInt16]]
}

/// Moving mean or median over `window` consecutive pairs (odd, centred) at each grid point.
///
/// `usable[t][k]` marks raw vectors that may contribute; `gate` marks where an output is
/// wanted. Outputs with fewer than `minSupport` contributing vectors are NaN. `leaveOut` lists
/// time offsets to skip (for cross-validation); each skipped non-zero offset is replaced by the
/// next one further out.
public func temporalSmooth(
    dx: [[Float]], dy: [[Float]], usable: [[Bool]], gate: [[Bool]],
    kernel: TemporalSmoothing.Kernel, window: Int, minSupport: Int, leaveOut: [Int] = []
) -> SmoothedSeries {
    precondition(window % 2 == 1, "temporal window must be odd")
    let nt = dx.count
    let cells = dx.first?.count ?? 0
    let half = window / 2
    var offsets: [Int] = leaveOut.contains(0) ? [] : [0]
    for sign in [-1, 1] {
        var (k, taken) = (1, 0)
        while taken < half {
            if !leaveOut.contains(sign * k) {
                offsets.append(sign * k)
                taken += 1
            }
            k += 1
        }
    }
    var out = SmoothedSeries(
        dx: Array(repeating: Array(repeating: .nan, count: cells), count: nt),
        dy: Array(repeating: Array(repeating: .nan, count: cells), count: nt),
        support: Array(repeating: Array(repeating: 0, count: cells), count: nt)
    )
    var bu: [Float] = []
    var bv: [Float] = []
    bu.reserveCapacity(window)
    bv.reserveCapacity(window)
    for k in 0..<cells {
        for t in 0..<nt where gate[t][k] {
            bu.removeAll(keepingCapacity: true)
            bv.removeAll(keepingCapacity: true)
            for o in offsets {
                let s = t + o
                guard s >= 0, s < nt, usable[s][k] else { continue }
                bu.append(dx[s][k])
                bv.append(dy[s][k])
            }
            out.support[t][k] = UInt16(bu.count)
            if bu.count >= max(minSupport, 1) {
                switch kernel {
                case .mean:
                    out.dx[t][k] = mean(bu)
                    out.dy[t][k] = mean(bv)
                case .median:
                    out.dx[t][k] = median(&bu)
                    out.dy[t][k] = median(&bv)
                }
            }
        }
    }
    return out
}

/// f32 accumulation, as in Rust's `iter().sum::<f32>()`.
private func mean(_ v: [Float]) -> Float {
    v.reduce(Float(0), +) / Float(v.count)
}
