import Foundation
import Testing
@testable import PIVKit

/// The CPU stages against Rust on the same inputs: these must match exactly, since both run the
/// same f32 arithmetic in the same order.
@Suite struct RustParityCPUTests {
    static let cases = (try? RustFixture.names()) ?? []

    @Test(arguments: cases)
    func validationMatchesRustOnEveryPass(_ name: String) throws {
        let fx = try RustFixture.load(name)
        for (i, pass) in fx.passes.enumerated() {
            var f = pass.raw
            validate(&f, config: fx.validationConfig, limits: ValidationLimits(maxDisplacement: fx.maxDisplacement))
            #expect(f.flags == pass.validated.flags, "\(name) pass \(i): flags")
            #expect(sameFloats(f.dx, pass.validated.dx), "\(name) pass \(i): dx")
            #expect(sameFloats(f.dy, pass.validated.dy), "\(name) pass \(i): dy")
        }
    }

    @Test func fixturesArePresent() {
        #expect(Self.cases.count >= 7)
    }
}
