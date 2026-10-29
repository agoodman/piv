import Foundation
@testable import PIVKit

/// Reference data written by `cargo run -p piv-core --example fixtures` (see
/// `rust/crates/piv-core/examples/fixtures.rs`).
struct RustFixture {
    struct Pass {
        let grid: Grid
        /// `[ax, ay, bx, by]` per grid cell.
        let origins: [SIMD4<Float>]
        let raw: VectorField
        let validated: VectorField
    }

    let name: String
    let height: Int, width: Int
    let frameA: [Float], frameB: [Float]
    let background: [Float]?
    let preparedA: [Float], preparedB: [Float]
    let meta: [String: Any]
    let passes: [Pass]

    var final: VectorField { passes.last!.validated }

    static let root: URL = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!
        .appendingPathComponent("cases")

    static func names() throws -> [String] {
        try JSONDecoder().decode([String].self, from: Data(contentsOf: root.appendingPathComponent("index.json")))
    }

    static func load(_ name: String) throws -> RustFixture {
        let dir = root.appendingPathComponent(name)
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("meta.json"))) as! [String: Any]
        func f32(_ file: Any?) throws -> [Float]? {
            guard let file = file as? String else { return nil }
            let d = try Data(contentsOf: dir.appendingPathComponent(file))
            return d.withUnsafeBytes { raw in
                (0..<d.count / 4).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
            }
        }
        func u8(_ file: Any?) throws -> [UInt8] {
            [UInt8](try Data(contentsOf: dir.appendingPathComponent(file as! String)))
        }
        func field(_ grid: Grid, _ m: [String: Any]) throws -> VectorField {
            VectorField(
                grid: grid, dx: try f32(m["dx"])!, dy: try f32(m["dy"])!,
                peakRatio: try f32(m["peak_ratio"])!,
                flags: try u8(m["flags"]).map(VectorFlags.init(rawValue:))
            )
        }
        let passes = try (meta["passes"] as! [[String: Any]]).map { p in
            let grid = try JSONDecoder().decode(Grid.self, from: JSONSerialization.data(withJSONObject: p["grid"]!))
            let o = try f32(p["origins"])!
            return Pass(
                grid: grid,
                origins: stride(from: 0, to: o.count, by: 4).map { SIMD4(o[$0], o[$0 + 1], o[$0 + 2], o[$0 + 3]) },
                raw: try field(grid, p["raw"] as! [String: Any]),
                validated: try field(grid, p["validated"] as! [String: Any])
            )
        }
        return RustFixture(
            name: name, height: meta["height"] as! Int, width: meta["width"] as! Int,
            frameA: try f32(meta["frame_a"])!, frameB: try f32(meta["frame_b"])!,
            background: try f32(meta["background"]),
            preparedA: try f32(meta["prepared_a"])!, preparedB: try f32(meta["prepared_b"])!,
            meta: meta, passes: passes
        )
    }

    /// The PIV settings the case was run with.
    var pivConfig: [String: Any] { meta["piv"] as! [String: Any] }
    var validationConfig: ValidationConfig {
        let v = pivConfig["validation"] as! [String: Any]
        var c = ValidationConfig()
        c.pprThreshold = Float(v["ppr_threshold"] as! Double)
        c.medianThreshold = Float(v["median_threshold"] as! Double)
        c.medianEps = Float(v["median_eps"] as! Double)
        c.replace = v["replace"] as! Bool
        return c
    }
    var maxDisplacement: Float { Float((pivConfig["windows"] as! [Int])[0]) / 2 }
}

/// Bitwise equality that treats NaN == NaN.
func sameFloats(_ a: [Float], _ b: [Float]) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { $0 == $1 || ($0.isNaN && $1.isNaN) }
}
