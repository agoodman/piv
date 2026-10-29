import Foundation

/// The summary `piv analyze` prints: how many vectors were usable, how far things moved, and
/// a hint for choosing the stride.
public struct RunSummary: Sendable {
    public var pairs: Int
    public var vectors: Int
    public var valid: Double
    public var rejected: Double
    public var replaced: Double
    public var noSignal: Double
    /// Percentiles of |displacement| over accepted vectors, px per pair.
    public var medianDisplacement: Float
    public var p95Displacement: Float
    public var p99Displacement: Float
    /// Median per-pair random error of one velocity component (neighbour-median estimate), px.
    public var noise: Float
    public var strideHint: String?

    public init(_ r: RunResult) {
        var (vectors, noSignal, rejected, replaced) = (0, 0, 0, 0)
        var mags: [Float] = []
        for f in r.raw {
            for k in f.flags.indices {
                vectors += 1
                let fl = f.flags[k]
                if fl.contains(.noSignal) {
                    noSignal += 1
                } else if !fl.isValid {
                    rejected += 1
                    if fl.contains(.replaced) { replaced += 1 }
                } else if vectors % 3 == 0 {
                    mags.append(hypotf(f.dx[k], f.dy[k]))
                }
            }
        }
        mags.sort()
        func pct(_ q: Double) -> Float { mags.isEmpty ? .nan : mags[Int((Double(mags.count - 1) * q).rounded())] }
        let n = Double(max(vectors, 1))
        pairs = r.pairs.count
        self.vectors = vectors
        valid = Double(vectors - noSignal - rejected) / n
        self.rejected = Double(rejected) / n
        self.replaced = Double(replaced) / n
        self.noSignal = Double(noSignal) / n
        medianDisplacement = pct(0.5)
        p95Displacement = pct(0.95)
        p99Displacement = pct(0.99)
        var noises = r.raw.map { f in
            neighbourNoise(u: f.dx, v: f.dy, usable: statsUsable(dx: f.dx, flags: f.flags), nx: f.grid.nx, ny: f.grid.ny)
        }.filter(\.isFinite)
        noises.sort()
        noise = noises.isEmpty ? .nan : noises[noises.count / 2]
        let p90 = pct(0.9)
        if p90.isFinite && p90 < 1 {
            strideHint = "Most motion is under 1 px per pair: a larger stride would improve the signal-to-noise ratio."
        } else if Double(rejected) > 0.3 * Double(max(vectors - noSignal, 1)) {
            strideHint = "Many vectors were rejected: try larger windows or a smaller stride."
        } else {
            strideHint = nil
        }
    }
}

/// The per-pair readout shown over the quiver video (as in the Rust overlay).
public struct ReadoutValues: Sendable {
    public var time: Double
    public var pair: Int
    public var pairs: Int
    public var vorticityMean: Float
    public var vorticityAbsMean: Float
    public var keTotal: (value: Double, unit: String)
    public var keMean: (value: Double, unit: String)
    public var coverage: Float

    public init(result r: RunResult, pair i: Int) {
        let s = r.stats[i]
        time = r.pairs[i].time
        pair = i
        pairs = r.pairs.count
        vorticityMean = s.vorticityMean
        vorticityAbsMean = s.vorticityAbsMean
        keTotal = r.energyUnits.keTotal(s.keSum)
        keMean = r.energyUnits.keMean(s.keMean)
        coverage = s.coverage
    }

    /// Number with about three significant digits; scientific outside [0.01, 10⁴).
    public static func format(_ x: Double) -> String {
        guard x.isFinite else { return "–" }
        let a = abs(x)
        if a == 0 { return "0" }
        if a < 0.01 || a >= 1e4 { return String(format: "%.2e", x) }
        let digits = max(0, 2 - Int(floor(log10(a))))
        return String(format: "%.\(digits)f", x)
    }
}
