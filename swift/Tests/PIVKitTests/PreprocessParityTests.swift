import Foundation
import Testing
@testable import PIVKit

/// GPU frame preparation against the Rust reference cases.
@Suite(.enabled(if: gpu != nil, "needs a Metal device"))
struct PreprocessParityTests {
    static let cases = (try? RustFixture.names()) ?? []
    let ctx = gpu!

    func preparer(_ fx: RustFixture) throws -> FramePreparer {
        let pre = fx.meta["preprocess"] as! [String: Any]
        var c = PreprocessConfig()
        c.smoothSigma = Float(pre["smooth_sigma"] as! Double)
        c.normalizeRadius = pre["normalize_radius"] as! Int
        c.normalizeEps = Float(pre["normalize_eps"] as! Double)
        let p = try FramePreparer(context: ctx, config: c, roi: ROI(x: 0, y: 0, width: fx.width, height: fx.height), channel: .luma)
        try p.setBackground(fx.background)
        return p
    }

    @Test func gaussianKernelMatchesRust() {
        let (k, r) = FramePreparer.gaussianKernel(sigma: 1)
        #expect(r == 3 && k.count == 7)
        #expect(abs(k.reduce(0, +) - 1) < 1e-6)
        #expect(FramePreparer.gaussianKernel(sigma: 0.2).1 == 1)
    }

    /// Prepared images within 1e-4 of Rust's (blur in f32 on both; local normalization in f32
    /// on the GPU vs f64 sums in Rust, divided by a small local std).
    @Test(arguments: cases)
    func preparedImagesMatchRust(_ name: String) async throws {
        let fx = try RustFixture.load(name)
        let p = try preparer(fx)
        for (raw, ref) in [(fx.frameA, fx.preparedA), (fx.frameB, fx.preparedB)] {
            let out = try await p.prepare(raw: raw).gpu.image.floats(raw.count)
            let worst = zip(out, ref).map { abs($0 - $1) }.max()!
            let scale = ref.map(abs).max()!
            if ProcessInfo.processInfo.environment["PIV_VERBOSE"] != nil {
                print("prep \(name): worst |Δ| \(worst) (max |value| \(scale))")
            }
            #expect(worst <= 1e-4 * max(scale, 1), "\(name): worst |Δ| \(worst)")
        }
    }

    /// Raw frames in, final field out: preparation on the GPU plus the multipass analysis.
    @Test(arguments: cases)
    func rawFramesToFieldMatchesRust(_ name: String) async throws {
        let fx = try RustFixture.load(name)
        let p = try preparer(fx)
        let a = try await p.prepare(raw: fx.frameA), b = try await p.prepare(raw: fx.frameB)
        let rp = RustParityGPUTests()
        let r = try await PIVAnalyzer(config: rp.config(fx), context: ctx).analyze(a.gpu, b.gpu)
        let ref = fx.final
        #expect(r.field.flags == ref.flags, "\(name): flags")
        var worst: Float = 0
        for k in ref.dx.indices where ref.dx[k].isFinite {
            worst = max(worst, abs(r.field.dx[k] - ref.dx[k]), abs(r.field.dy[k] - ref.dy[k]))
        }
        if ProcessInfo.processInfo.environment["PIV_VERBOSE"] != nil { print("raw→field \(name): worst |Δd| \(worst)") }
        #expect(worst < 1e-3, "\(name): worst |Δd| \(worst)")
    }
}
