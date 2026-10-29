import CoreGraphics
import Foundation
import Testing
@testable import PIVKit

@Suite(.enabled(if: gpu != nil, "needs a Metal device"))
struct RenderTests {
    let ctx = gpu!

    /// As the Rust `draws_arrows_where_vectors_exist`: arrows in every cell with a vector, none
    /// where the vector is missing.
    @Test func drawsArrowsWhereVectorsExist() async throws {
        let grid = try Grid(height: 64, width: 64, window: 32, overlap: 0)
        var f = VectorField(grid: grid)
        f.dx = [4, 4, 4, .nan]
        f.dy = [0, 0, 0, .nan]
        f.flags = [[], [], [], .noSignal]
        let raw = [Float](repeating: 0, count: 64 * 64)
        let auto = QuiverStyle.auto(raw: raw, field: f)
        #expect(abs(auto.arrowScale - 0.6 * 32 / 4) < 1e-4)
        var style = auto
        style.arrowScale = 2
        let img = try await QuiverRenderer(context: ctx).image(raw: try ctx.makeBuffer(raw), width: 64, height: 64, field: f, style: style)
        let px = pixels(img)
        func lit(_ x0: Int, _ y0: Int) -> Bool {
            (x0..<x0 + 32).contains { x in (y0..<y0 + 32).contains { y in px[(y * 64 + x) * 4 ..< (y * 64 + x) * 4 + 3].contains { $0 > 0 } } }
        }
        #expect(lit(0, 0) && lit(32, 0) && lit(0, 32))
        #expect(!lit(32, 32))
    }

    @Test func replacedVectorsAreMagentaAndBackgroundIsGray() async throws {
        let grid = try Grid(height: 32, width: 32, window: 32, overlap: 0)
        let f = VectorField(grid: grid, dx: [6], dy: [0], peakRatio: [2], flags: [[.medianOutlier, .replaced]])
        let raw = [Float](repeating: 0.5, count: 32 * 32)
        let style = QuiverStyle(arrowScale: 2, colorMax: 10, gain: 1)
        let px = pixels(try await QuiverRenderer(context: ctx).image(raw: try ctx.makeBuffer(raw), width: 32, height: 32, field: f, style: style))
        // Background: floor(0.5 · 180) = 90 in every channel (BGRA bytes).
        #expect(Array(px[0..<3]) == [90, 90, 90])
        // The shaft (y = 15.5, between pixel rows 15 and 16, so each is half covered) is magenta
        // blended over the gray: blue and red well above green.
        let k = (15 * 32 + 20) * 4
        let (b, g, r) = (Int(px[k]), Int(px[k + 1]), Int(px[k + 2]))
        #expect(b > g + 60 && r > g + 60, "BGR \(b), \(g), \(r)")
    }

    func pixels(_ img: CGImage) -> [UInt8] {
        let data = img.dataProvider!.data! as Data
        return [UInt8](data)
    }
}
