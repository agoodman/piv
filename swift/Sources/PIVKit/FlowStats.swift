import Foundation

/// Flow-wide statistics of one vector field: mean vorticity and kinetic energy. Port of
/// `piv_core::flowstats`.
///
/// Inputs are velocity components on a regular grid (any consistent units; px/s in practice)
/// and a mask of the vectors to use. Image axes point right and *down*; vorticity is reported
/// with the usual y-up sign, so **positive means counter-clockwise as seen on screen**.
public struct FlowStats: Hashable, Sendable, Codable {
    /// Mean vorticity (1/s for px/s input), + = counter-clockwise on screen. NaN if undefined.
    public var vorticityMean: Float
    /// Mean |vorticity|. NaN if undefined.
    public var vorticityAbsMean: Float
    /// ½⟨u² + v²⟩ over the usable vectors (input units²). NaN if there are none.
    public var keMean: Float
    /// ½ Σ (u² + v²) · step² over the usable vectors (input units² × px²); 0 if there are none.
    public var keSum: Float
    /// Usable vectors / grid cells.
    public var coverage: Float
    /// Usable vectors.
    public var nUsable: Int
    /// Points where vorticity was defined.
    public var nVorticity: Int
}

/// One derivative along an axis at `i` of a line of `n` samples, spacing `h`. Central where
/// both neighbours are usable, one-sided where only one is (the centre must be usable).
private func derivative(_ i: Int, _ n: Int, _ h: Float, _ at: (Int) -> Float?) -> Float? {
    guard let c = at(i) else { return nil }
    let lo = i > 0 ? at(i - 1) : nil
    let hi = i + 1 < n ? at(i + 1) : nil
    switch (lo, hi) {
    case let (l?, r?): return (r - l) / (2 * h)
    case let (l?, nil): return (c - l) / h
    case let (nil, r?): return (r - c) / h
    case (nil, nil): return nil
    }
}

/// Vorticity ω = ∂u/∂y − ∂v/∂x in image coordinates (y down), which is the y-up ∂v/∂x − ∂u/∂y,
/// so positive is counter-clockwise on screen. Arrays are row-major `[ny][nx]`; `step` is the
/// grid spacing in px. NaN where a derivative along either axis has no usable neighbour.
public func vorticity(
    u: [Float], v: [Float], usable: [Bool], nx: Int, ny: Int, step: Float
) -> [Float] {
    func get(_ a: [Float], _ iy: Int, _ ix: Int) -> Float? {
        let x = a[iy * nx + ix]
        return usable[iy * nx + ix] && x.isFinite ? x : nil
    }
    var w = [Float](repeating: .nan, count: nx * ny)
    for iy in 0..<ny {
        for ix in 0..<nx {
            let duDy = derivative(iy, ny, step) { get(u, $0, ix) }
            let dvDx = derivative(ix, nx, step) { get(v, iy, $0) }
            if let a = duDy, let b = dvDx { w[iy * nx + ix] = a - b }
        }
    }
    return w
}

/// Mean vorticity and kinetic energy of one field. See ``FlowStats``.
public func flowStats(
    u: [Float], v: [Float], usable: [Bool], nx: Int, ny: Int, step: Float
) -> FlowStats {
    let w = vorticity(u: u, v: v, usable: usable, nx: nx, ny: ny, step: step)
    var (ws, wa, nw) = (0.0, 0.0, 0)
    for x in w where x.isFinite {
        ws += Double(x)
        wa += Double(abs(x))
        nw += 1
    }
    var (e, n) = (0.0, 0)
    for k in u.indices where usable[k] && u[k].isFinite && v[k].isFinite {
        let (a, b) = (Double(u[k]), Double(v[k]))
        e += 0.5 * (a * a + b * b)
        n += 1
    }
    func mean(_ s: Double, _ n: Int) -> Float { n > 0 ? Float(s / Double(n)) : .nan }
    return FlowStats(
        vorticityMean: mean(ws, nw),
        vorticityAbsMean: mean(wa, nw),
        keMean: mean(e, n),
        keSum: Float(e * Double(step) * Double(step)),
        coverage: Float(n) / Float(max(u.count, 1)),
        nUsable: n,
        nVorticity: nw
    )
}

/// Estimates the random error of one vector component (same units as `u`, `v`) from how much
/// each usable vector differs from the median of its usable 8-neighbours. Real small-scale flow
/// also counts as "error", so this is an upper bound. NaN with too few points.
public func neighbourNoise(u: [Float], v: [Float], usable: [Bool], nx: Int, ny: Int) -> Float {
    var resid: [Float] = []
    var nb: [Float] = []
    nb.reserveCapacity(8)
    for a in [u, v] {
        for iy in 0..<ny {
            for ix in 0..<nx {
                let k = iy * nx + ix
                guard usable[k], a[k].isFinite else { continue }
                nb.removeAll(keepingCapacity: true)
                for jy in max(iy - 1, 0)..<min(iy + 2, ny) {
                    for jx in max(ix - 1, 0)..<min(ix + 2, nx) {
                        let j = jy * nx + jx
                        if j != k, usable[j], a[j].isFinite { nb.append(a[j]) }
                    }
                }
                guard nb.count >= 5 else { continue }
                nb.sort { totalOrderLess($0, $1) }
                resid.append(abs(a[k] - nb[nb.count / 2]))
            }
        }
    }
    guard resid.count >= 20 else { return .nan }
    resid.sort { totalOrderLess($0, $1) }
    let mad = resid[resid.count / 2]
    // MAD → σ for a normal residual; the neighbour median itself carries ≈ π/(2·8) of the
    // variance of one vector.
    return 1.4826 * mad / (1 + Float.pi / 16).squareRoot()
}

/// Vectors that count toward the statistics: accepted or interpolated (`replaced`), and finite.
public func statsUsable(dx: [Float], flags: [VectorFlags]) -> [Bool] {
    zip(dx, flags).map { d, f in d.isFinite && (f.isValid || f.contains(.replaced)) }
}

/// Statistics of one pair from displacements (px), in px/s units.
public func pairStats(_ field: VectorField, dt: Double) -> FlowStats {
    let k = Float(1 / dt)
    return flowStats(
        u: field.dx.map { $0 * k }, v: field.dy.map { $0 * k },
        usable: statsUsable(dx: field.dx, flags: field.flags),
        nx: field.grid.nx, ny: field.grid.ny, step: Float(field.grid.step)
    )
}

/// Air at 20 °C, kg/m³.
public let airDensity = 1.204

/// Converts the px-based kinetic energy to the most physical units the run supports.
public struct EnergyUnits: Hashable, Sendable {
    public var metresPerPixel: Double?
    public var thicknessMetres: Double?
    /// kg/m³
    public var density: Double

    public init(metresPerPixel: Double?, thicknessMetres: Double?, density: Double = airDensity) {
        self.metresPerPixel = metresPerPixel
        self.thicknessMetres = thicknessMetres
        self.density = density
    }

    /// ½⟨u² + v²⟩ (px²/s²) → (value, unit): m²/s² with a scale.
    public func keMean(_ px2: Float) -> (value: Double, unit: String) {
        if let k = metresPerPixel { (Double(px2) * k * k, "m²/s²") } else { (Double(px2), "px²/s²") }
    }

    /// ½ Σ (u² + v²) · A (px⁴/s²) → (value, unit): ×ρ per metre of sheet depth (mJ/m) with a
    /// scale, × thickness (µJ) with a thickness too.
    public func keTotal(_ px4: Float) -> (value: Double, unit: String) {
        switch (metresPerPixel, thicknessMetres) {
        case let (k?, t?): (Double(px4) * pow(k, 4) * density * t * 1e6, "µJ")
        case let (k?, nil): (Double(px4) * pow(k, 4) * density * 1e3, "mJ/m")
        default: (Double(px4), "px⁴/s²")
        }
    }

    /// Short description of what ``keTotal(_:)`` means.
    public var keTotalLabel: String {
        switch (metresPerPixel, thicknessMetres) {
        case (_?, _?): "in-plane KE in the measured slab"
        case (_?, nil): "in-plane KE per metre of sheet depth"
        default: "in-plane KE per unit density and depth"
        }
    }
}
