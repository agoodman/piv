import Foundation
import Testing
@testable import PIVKit

/// The GPU correlation against Rust, stage by stage, on the reference cases from
/// `rust/crates/piv-core/examples/fixtures.rs`.
@Suite(.enabled(if: gpu != nil, "needs a Metal device"))
struct RustParityGPUTests {
    static let cases = (try? RustFixture.names()) ?? []
    let ctx = gpu!

    func frames(_ fx: RustFixture) throws -> (GPUFrame, GPUFrame) {
        func stats(_ raw: [Float]) -> [Float] {
            guard let bg = fx.background else { return raw }
            return zip(raw, bg).map { max($0 - $1, 0) }
        }
        return (
            try ctx.makeFrame(image: fx.preparedA, stats: stats(fx.frameA), width: fx.width, height: fx.height),
            try ctx.makeFrame(image: fx.preparedB, stats: stats(fx.frameB), width: fx.width, height: fx.height)
        )
    }

    var config: (RustFixture) -> PIVConfig {
        { fx in
            var c = PIVConfig()
            let p = fx.pivConfig
            c.windows = p["windows"] as! [Int]
            c.overlap = Float(p["overlap"] as! Double)
            c.finalIterations = p["final_iterations"] as! Int
            let m = p["mask"] as! [String: Any]
            c.mask = MaskConfig(minMean: Float(m["min_mean"] as! Double), minStd: Float(m["min_std"] as! Double))
            c.validation = fx.validationConfig
            return c
        }
    }

    /// Each pass, given Rust's predictor: same origins (exactly), and raw correlation peaks
    /// within 1e-4 px (f32 rounding; measured ≤ 3e-6) with the same NO_SIGNAL decisions.
    @Test(arguments: cases, [CorrelationPath.auto, .batched, .mpsGraph])
    func rawPeaksMatchRustPassByPass(_ name: String, _ path: CorrelationPath) async throws {
        let fx = try RustFixture.load(name)
        let (a, b) = try frames(fx)
        let cfg = config(fx)
        let correlator = Correlator(context: ctx)
        var predictor: VectorField?
        for (i, pass) in fx.passes.enumerated() {
            let origins = PIVAnalyzer.placeWindows(grid: pass.grid, width: fx.width, height: fx.height, predictor: predictor)
            #expect(origins == pass.origins, "\(name) pass \(i): origins")
            let peaks = try await correlator.correlate(a: a, b: b, n: pass.grid.window, origins: pass.origins, mask: cfg.mask, path: path)
            let f = PIVAnalyzer.assemble(grid: pass.grid, origins: pass.origins, peaks: peaks)
            #expect(f.flags == pass.raw.flags, "\(name) pass \(i): flags")
            var worst: Float = 0, worstPPR: Float = 0
            for k in f.dx.indices where f.dx[k].isFinite {
                worst = max(worst, abs(f.dx[k] - pass.raw.dx[k]), abs(f.dy[k] - pass.raw.dy[k]))
                worstPPR = max(worstPPR, abs(f.peakRatio[k] / pass.raw.peakRatio[k] - 1))
            }
            if ProcessInfo.processInfo.environment["PIV_VERBOSE"] != nil {
                print("parity \(name) \(path) pass \(i) n=\(pass.grid.window): worst |Δd| \(worst) px, worst PPR rel \(worstPPR), vectors \(f.dx.filter(\.isFinite).count)/\(f.dx.count)")
            }
            #expect(worst < 1e-4, "\(name) pass \(i) (n=\(pass.grid.window)): worst |Δd| \(worst)")
            #expect(worstPPR < 1e-5, "\(name) pass \(i): worst PPR rel. diff \(worstPPR)")
            predictor = pass.validated
        }
    }

    /// GPU window placement from Rust's predictor reproduces Rust's origins bit for bit.
    @Test(arguments: cases)
    func gpuPlacementMatchesRustExactly(_ name: String) async throws {
        let fx = try RustFixture.load(name)
        let analyzer = try PIVAnalyzer(config: config(fx), context: ctx)
        var predictor: VectorField?
        for (i, pass) in fx.passes.enumerated() {
            let o = try await analyzer.placeOnGPU(grid: pass.grid, width: fx.width, height: fx.height, predictor: predictor)
            #expect(o == pass.origins, "\(name) pass \(i)")
            predictor = pass.validated
        }
    }

    /// GPU validation of Rust's raw fields reproduces Rust's validated fields bit for bit.
    @Test(arguments: cases)
    func gpuValidationMatchesRustExactly(_ name: String) async throws {
        let fx = try RustFixture.load(name)
        let analyzer = try PIVAnalyzer(config: config(fx), context: ctx)
        for (i, pass) in fx.passes.enumerated() {
            let v = try await analyzer.validateOnGPU(pass.raw)
            #expect(v.flags == pass.validated.flags, "\(name) pass \(i): flags")
            #expect(sameFloats(v.dx, pass.validated.dx) && sameFloats(v.dy, pass.validated.dy), "\(name) pass \(i): values")
        }
    }

    /// A traced run returns every pass, and its final field is the untraced result.
    @Test func tracedRunIsConsistent() async throws {
        let fx = try RustFixture.load("fog_default")
        let (a, b) = try frames(fx)
        let analyzer = try PIVAnalyzer(config: config(fx), context: ctx)
        let (r, traces) = try await analyzer.analyzeTraced(a, b)
        #expect(traces.count == fx.passes.count)
        #expect(traces.last!.validated == r.field)
        let plain = try await analyzer.analyze(a, b)
        #expect(plain.field == r.field && plain.summary == r.summary)
        #expect(r.summary.noSignal == fx.final.flags.filter { $0.contains(.noSignal) }.count)
    }

    /// The whole multi-pass analysis: final fields within 1e-4 px RMS, flags identical.
    @Test(arguments: cases)
    func finalFieldMatchesRust(_ name: String) async throws {
        let fx = try RustFixture.load(name)
        let (a, b) = try frames(fx)
        let r = try await PIVAnalyzer(config: config(fx), context: ctx).analyze(a, b)
        let ref = fx.final
        let same = zip(r.field.flags, ref.flags).filter { $0 == $1 }.count
        #expect(same == ref.flags.count, "\(name): \(same)/\(ref.flags.count) flags equal")
        var sum: Float = 0, n = 0
        for k in ref.dx.indices where ref.dx[k].isFinite && r.field.dx[k].isFinite {
            sum += pow(r.field.dx[k] - ref.dx[k], 2) + pow(r.field.dy[k] - ref.dy[k], 2)
            n += 1
        }
        let rms = (sum / Float(max(n, 1))).squareRoot()
        if ProcessInfo.processInfo.environment["PIV_VERBOSE"] != nil {
            print("parity \(name) final: RMS |Δd| \(rms) px over \(n) vectors, flags equal \(same)/\(ref.flags.count)")
        }
        #expect(rms < 1e-4, "\(name): RMS |Δd| \(rms) over \(n) vectors")
    }
}
