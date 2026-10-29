import Foundation
import Testing
@testable import PIVKit

private let noLimit = ValidationLimits(maxDisplacement: .infinity)

private func uniformField(_ u: Float, _ v: Float) throws -> VectorField {
    var f = VectorField(grid: try Grid(height: 160, width: 160, window: 32, overlap: 0.5))
    f.dx = Array(repeating: u, count: f.grid.count)
    f.dy = Array(repeating: v, count: f.grid.count)
    f.peakRatio = Array(repeating: 5, count: f.grid.count)
    f.flags = Array(repeating: [], count: f.grid.count)
    return f
}

@Suite struct ValidationTests {
    @Test func catchesAndReplacesInjectedOutliers() throws {
        var f = try uniformField(2, -1)
        for i in f.dx.indices { f.dx[i] += 0.05 * (Float(i * 7 % 5) - 2) }
        let outliers = [(2, 3), (5, 5), (8, 0)]
        for (iy, ix) in outliers {
            f.dx[f.index(ix, iy)] = 9
            f.dy[f.index(ix, iy)] = 4
        }
        let s = validate(&f, config: ValidationConfig(), limits: noLimit)
        #expect(s.medianOutliers == outliers.count)
        #expect(s.replaced == outliers.count)
        for (iy, ix) in outliers {
            let k = f.index(ix, iy)
            #expect(f.flags[k].contains([.medianOutlier, .replaced]))
            #expect(abs(f.dx[k] - 2) < 0.2 && abs(f.dy[k] + 1) < 1e-6)
        }
    }

    @Test func lowPeakRatioIsFlaggedAndNoSignalUntouched() throws {
        var f = try uniformField(1, 1)
        f.peakRatio[f.index(4, 4)] = 1.05
        f.flags[0] = .noSignal
        f.dx[0] = .nan
        f.dy[0] = .nan
        let s = validate(&f, config: ValidationConfig(), limits: noLimit)
        #expect((s.lowPPR, s.noSignal, s.medianOutliers) == (1, 1, 0))
        #expect(f.flags[f.index(4, 4)] == [.lowPPR, .replaced])
        #expect(f.flags[0] == .noSignal)
        #expect(f.dx[0].isNaN)
    }

    @Test func outOfRangeVectorsAreFlaggedAndReplaced() throws {
        var f = try uniformField(1, 0.5)
        f.dx[f.index(3, 3)] = 40
        let s = validate(&f, config: ValidationConfig(), limits: ValidationLimits(maxDisplacement: 32))
        #expect(s.outOfRange == 1)
        #expect(f.flags[f.index(3, 3)].contains([.outOfRange, .replaced]))
        #expect(f.dx[f.index(3, 3)] == 1)
    }

    @Test func medianHandlesEvenAndOdd() {
        var a: [Float] = [3, 1, 2]
        var b: [Float] = [4, 1, 2, 3]
        #expect(median(&a) == 2)
        #expect(median(&b) == 2.5)
    }
}

@Suite struct FlowStatsTests {
    let n = 21
    let step: Float = 16

    /// Velocity from a function of image position (x right, y down), all usable.
    func field(_ f: (Float, Float) -> (Float, Float)) -> (u: [Float], v: [Float], ok: [Bool]) {
        var u: [Float] = [], v: [Float] = []
        for iy in 0..<n {
            for ix in 0..<n {
                let p = f(Float(ix) * step, Float(iy) * step)
                u.append(p.0)
                v.append(p.1)
            }
        }
        return (u, v, Array(repeating: true, count: n * n))
    }

    func stats(_ u: [Float], _ v: [Float], _ ok: [Bool]) -> FlowStats {
        flowStats(u: u, v: v, usable: ok, nx: n, ny: n, step: step)
    }

    @Test func solidBodyRotationCounterClockwiseOnScreenIsPositive() {
        let om: Float = 3
        let c = Float(n - 1) * step / 2
        let (u, v, ok) = field { x, y in (om * (y - c), -om * (x - c)) }
        let w = vorticity(u: u, v: v, usable: ok, nx: n, ny: n, step: step)
        #expect(w.allSatisfy { abs($0 - 2 * om) <= 1e-4 })
        let s = stats(u, v, ok)
        #expect(abs(s.vorticityMean - 2 * om) <= 1e-4)
        #expect(abs(s.vorticityAbsMean - 2 * om) <= 1e-4)
        #expect(s.nVorticity == n * n)
    }

    @Test func uniformTranslation() {
        let (u, v, ok) = field { _, _ in (3, -4) }
        let s = stats(u, v, ok)
        #expect(s.vorticityMean == 0 && s.vorticityAbsMean == 0)
        #expect(abs(s.keMean - 12.5) <= 1e-5)
        #expect(abs(s.keSum - 12.5 * Float(n * n) * step * step) <= 1)
        #expect(s.coverage == 1)
    }

    @Test func shear() {
        let (u, v, ok) = field { _, y in (0.5 * y, 0) }
        #expect(abs(stats(u, v, ok).vorticityMean - 0.5) <= 1e-5)
    }

    @Test func oppositeVorticesCancelInTheSignedMean() {
        let l = Float(n) * step
        let (u, v, ok) = field { x, _ in (0, sin(2 * .pi * x / l) * 10) }
        let s = stats(u, v, ok)
        #expect(abs(s.vorticityMean) < 0.1 * s.vorticityAbsMean)
        #expect(abs(s.vorticityAbsMean - 40 / l) <= 0.1 * 40 / l)
    }

    @Test func holesAndEdges() {
        let (u, v, okAll) = field { x, y in (y * 0.1, x * 0.2) }
        var ok = okAll
        ok[5 * n + 5] = false
        let w = vorticity(u: u, v: v, usable: ok, nx: n, ny: n, step: step)
        #expect(w[5 * n + 5].isNaN)
        for (iy, ix) in [(4, 5), (6, 5), (5, 4), (5, 6), (0, 0), (n - 1, n - 1)] {
            #expect(abs(w[iy * n + ix] + 0.1) <= 1e-5, "\(iy),\(ix): \(w[iy * n + ix])")
        }
        var lonely = Array(repeating: false, count: n * n)
        lonely[3 * n + 3] = true
        var s = stats(u, v, lonely)
        #expect(s.nVorticity == 0 && s.vorticityMean.isNaN && s.nUsable == 1)
        var un = u
        un[0] = .nan
        s = stats(un, v, ok)
        #expect(s.nUsable == n * n - 2)
    }

    @Test func emptyField() {
        let (u, v, _) = field { _, _ in (1, 1) }
        let s = stats(u, v, Array(repeating: false, count: n * n))
        #expect(s.keMean.isNaN && s.vorticityMean.isNaN)
        #expect((s.keSum, s.coverage, s.nUsable) == (0, 0, 0))
    }

    @Test func neighbourNoiseRecoversSigma() {
        var state: UInt64 = 12345
        func uniform() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Double(state >> 11) / Double(UInt64(1) << 53))
        }
        func normal() -> Float {
            let (a, b) = (max(uniform(), 1e-9), uniform())
            return (-2 * log(a)).squareRoot() * cos(2 * .pi * b)
        }
        let m = 80
        let u = (0..<m * m).map { _ in 2 * normal() }
        let v = (0..<m * m).map { _ in 2 * normal() }
        let s = neighbourNoise(u: u, v: v, usable: Array(repeating: true, count: m * m), nx: m, ny: m)
        #expect(abs(s - 2) < 0.15, "\(s)")
    }

    @Test func energyUnits() {
        let none = EnergyUnits(metresPerPixel: nil, thicknessMetres: nil)
        #expect(none.keTotal(5).unit == "px⁴/s²")
        let scaled = EnergyUnits(metresPerPixel: 1e-3, thicknessMetres: nil, density: 1)
        #expect(abs(scaled.keTotal(1e12).value / 1e3 - 1) < 1e-6)
        #expect(scaled.keMean(4).unit == "m²/s²" && abs(scaled.keMean(4).value - 4e-6) < 1e-18)
        let slab = EnergyUnits(metresPerPixel: 1e-3, thicknessMetres: 1e-3, density: 1)
        #expect(slab.keTotal(1e12).unit == "µJ" && abs(slab.keTotal(1e12).value / 1e3 - 1) < 1e-6)
    }
}

@Suite struct TemporalSmoothingTests {
    @Test func medianAndMeanOverOddWindowWithSupport() {
        // One grid cell, 7 pairs; pair 3 is an outlier and pair 5 is unusable.
        let dx: [[Float]] = [[1], [1], [1], [10], [1], [99], [1]]
        let dy: [[Float]] = dx.map { $0.map { -$0 } }
        var usable = Array(repeating: [true], count: 7)
        usable[5] = [false]
        let gate = Array(repeating: [true], count: 7)
        let med = temporalSmooth(dx: dx, dy: dy, usable: usable, gate: gate, kernel: .median, window: 3, minSupport: 2)
        #expect(Array(med.dx.map { $0[0] }.prefix(6)) == [1, 1, 1, 1, 5.5, 1])
        #expect(med.dx[3][0] == 1 && med.dy[3][0] == -1)
        #expect(med.support.map { $0[0] } == [2, 3, 3, 3, 2, 2, 1])
        #expect(med.dx[6][0].isNaN) // support 1 < 2
        let mean = temporalSmooth(dx: dx, dy: dy, usable: usable, gate: gate, kernel: .mean, window: 3, minSupport: 1)
        #expect(mean.dx[2][0] == 4)
    }

    @Test func gateBlocksOutputAndLeaveOutSkipsNeighbours() {
        let dx: [[Float]] = (0..<9).map { [Float($0)] }
        let usable = Array(repeating: [true], count: 9)
        var gate = Array(repeating: [true], count: 9)
        gate[4] = [false]
        let s = temporalSmooth(dx: dx, dy: dx, usable: usable, gate: gate, kernel: .mean, window: 3, minSupport: 1)
        #expect(s.dx[4][0].isNaN && s.support[4][0] == 0)
        // Leaving out 0 and ±1 at window 3 uses ±2: the mean of t−2 and t+2 is t.
        let l = temporalSmooth(dx: dx, dy: dx, usable: usable, gate: usable, kernel: .mean, window: 3, minSupport: 1, leaveOut: [0, -1, 1])
        #expect(l.dx[4][0] == 4 && l.support[4][0] == 2)
    }

    @Test func applyMatchesPivSmoothConventions() throws {
        let grid = try Grid(height: 32, width: 32, window: 32, overlap: 0)
        let fields = (0..<5).map { t in
            VectorField(grid: grid, dx: [Float(t)], dy: [0], peakRatio: [3],
                        flags: [t == 2 ? [.lowPPR, .replaced] : []])
        }
        let (out, support) = TemporalSmoothing(window: 3, kernel: .mean).apply(to: fields)
        #expect(TemporalSmoothing(window: 9).resolvedMinSupport == 3)
        #expect(TemporalSmoothing(window: 3).resolvedMinSupport == 1)
        // Pair 2 is rejected (not usable) but seeded, so it gets the mean of 1 and 3.
        #expect(out[2].dx[0] == 2 && support[2][0] == 2 && out[2].flags[0] == [])
        #expect(out[0].dx[0] == 0.5)
    }
}
