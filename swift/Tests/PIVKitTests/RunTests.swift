import CoreVideo
import Foundation
import Testing
@testable import PIVKit

@Suite struct PairPlanTests {
    /// The same roles as `piv analyze`: pairs (i, i + stride) for i in [start, end − stride)
    /// stepping by pairStep.
    @Test func rolesAndStarts() {
        let p = PairPlan(start: 10, end: 30, stride: 5, pairStep: 3)
        #expect(p.pairStarts == [10, 13, 16, 19, 22])
        #expect(p.isA(10) && p.isA(22) && !p.isA(25) && !p.isA(11))
        #expect(p.isB(15) && p.isB(27) && !p.isB(16) && !p.isB(30))
    }
}

@Suite struct NpyTests {
    @Test func headerIsAlignedAndParsable() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("pivkit-\(UUID()).npy")
        defer { try? FileManager.default.removeItem(at: url) }
        try Npy.write([Float](repeating: 1.5, count: 6), shape: [2, 3], dtype: .f4, to: url)
        let d = try Data(contentsOf: url)
        #expect(Array(d.prefix(8)) == [0x93] + Array("NUMPY".utf8) + [1, 0])
        let hlen = Int(d[8]) | Int(d[9]) << 8
        #expect((10 + hlen) % 64 == 0)
        let header = String(decoding: d[10..<(10 + hlen)], as: UTF8.self)
        #expect(header.contains("'descr': '<f4'") && header.contains("'shape': (2, 3)") && header.hasSuffix("\n"))
        #expect(d.count == 10 + hlen + 24)
        let one = try Npy.data(shape: [4], dtype: .u1, body: Data([1, 2, 3, 4]))
        #expect(String(decoding: one, as: UTF8.self).contains("'shape': (4,)"))
    }
}

/// Against the sample clips, when `PIV_CLIPS` points at the `clips/` directory.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["PIV_CLIPS"] != nil, "set PIV_CLIPS to the clips directory"))
struct ClipTests {
    var c0141: URL { URL(fileURLWithPath: ProcessInfo.processInfo.environment["PIV_CLIPS"]!).appendingPathComponent("C0141.MP4") }

    @Test func probeC0141() async throws {
        let info = try await MediaInfo.probe(c0141)
        #expect((info.width, info.height, info.frameCount, info.bitDepth) == (1920, 1080, 3720, 8))
        #expect(info.containerRate == (try FrameRate(50)))
        #expect(info.cameraRate?.rate == (try FrameRate(1000)))
        #expect(info.cameraRate?.camera == "Sony DSC-RX100M5A")
    }

    /// Starting a decode mid-stream yields the same pictures, with the same indices, as
    /// decoding through from earlier (AVAssetReader emits a clipped extra sample if the range
    /// starts before a frame's timestamp).
    @Test func seekingIsFrameAccurate() async throws {
        let info = try await MediaInfo.probe(c0141)
        func checksums(_ range: Range<Int>) async throws -> [Int: UInt64] {
            let d = try await VideoDecoder(info: info, range: range, channel: .luma)
            var out: [Int: UInt64] = [:]
            while let f = try await d.nextFrame() {
                #expect(out[f.index] == nil, "index \(f.index) repeated")
                let pb = f.pixelBuffer
                CVPixelBufferLockBaseAddress(pb, .readOnly)
                let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self)
                let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
                var s: UInt64 = 0
                for y in Swift.stride(from: 0, to: 1080, by: 7) { for x in Swift.stride(from: 0, to: 1920, by: 5) { s = s &* 31 &+ UInt64(base[y * stride + x]) } }
                CVPixelBufferUnlockBaseAddress(pb, .readOnly)
                out[f.index] = s
            }
            return out
        }
        let through = try await checksums(190..<215)
        for start in [195, 200, 207] {
            let seeked = try await checksums(start..<(start + 6))
            #expect(Set(seeked.keys) == Set(start..<(start + 6)))
            for (i, s) in seeked { #expect(s == through[i], "frame \(i) after seeking to \(start)") }
        }
    }

    @Test func shortRunEndToEnd() async throws {
        var req = AnalysisRequest(source: c0141)
        req.frameRange = 200..<220
        req.stride = 5
        req.background = BackgroundSpec(mode: .min, frames: 4)
        req.temporalSmoothing = TemporalSmoothing(window: 3)
        let r = try await analyzeRun(req, context: gpu!)
        #expect(r.pairs.count == 15 && r.pairs.first?.frameA == 200 && r.pairs.last?.frameB == 219)
        #expect(abs(r.dt - 0.005) < 1e-12)
        #expect(r.raw.allSatisfy { $0.grid == r.grid } && r.smoothed?.count == 15)
        #expect(r.raw.map(\.validCount).reduce(0, +) > 0)
    }
}
