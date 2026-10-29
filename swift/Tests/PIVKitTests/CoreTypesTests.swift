import Foundation
import Testing
@testable import PIVKit

@Suite struct FrameRateTests {
    func rate(_ s: String) -> FrameRate { try! FrameRate(parsing: s) }
    func r(_ n: UInt64, _ d: UInt64 = 1) -> FrameRate { try! FrameRate(n, d) }

    @Test func parsesIntegerAndCameraStrings() {
        #expect(rate("240") == r(240))
        #expect(rate("1000p") == r(1000))
        #expect(rate(" 50P ") == r(50))
        #expect(rate("120fps") == r(120))
    }

    @Test func parsesRationalsAndReduces() {
        #expect(rate("240000/1001") == r(240000, 1001))
        #expect(rate("50000/1000") == r(50))
    }

    @Test func snapsNTSCDecimals() {
        #expect(rate("239.76") == r(240000, 1001))
        #expect(rate("29.97") == r(30000, 1001))
        #expect(rate("23.98") == r(24000, 1001))
        #expect(rate("119.88p") == r(120000, 1001))
    }

    @Test func keepsOtherDecimalsExact() {
        #expect(rate("12.5") == r(25, 2))
        #expect(rate("1000.0") == r(1000))
    }

    @Test(arguments: ["", "0", "abc", "-5", "1/0", "0/7", "1.", "nan"])
    func rejectsGarbage(_ s: String) {
        #expect(throws: PIVError.self) { try FrameRate(parsing: s) }
    }

    @Test func dtScalesWithStrideAndRate() {
        #expect(abs(rate("1000").dt(stride: 1) - 1e-3) < 1e-15)
        #expect(abs(rate("1000").dt(stride: 3) - 3e-3) < 1e-15)
        #expect(abs(rate("239.76").dt(stride: 2) - 2 * 1001 / 240000) < 1e-15)
        #expect(abs(rate("50").time(ofFrame: 3720) - 74.4) < 1e-9)
    }
}

@Suite struct GridTests {
    @Test func halfOverlapGridOnFullHD() throws {
        let g = try Grid(height: 1080, width: 1920, window: 32, overlap: 0.5)
        #expect(g.step == 16)
        #expect((g.nx, g.ny) == (119, 66))
        #expect((g.x0, g.y0) == (0, 4))
        #expect(g.centerX(0) == 15.5)
        let o = g.windowOrigin(g.nx - 1, g.ny - 1)
        #expect(o.x + 32 <= 1920 && o.y + 32 <= 1080)
    }

    @Test func noOverlapAndExactFit() throws {
        let g = try Grid(height: 64, width: 64, window: 32, overlap: 0)
        #expect((g.nx, g.ny, g.step, g.x0) == (2, 2, 32, 0))
        #expect(g.xs == [15.5, 47.5])
    }

    @Test func rejectsBadInputs() {
        #expect(throws: PIVError.self) { try Grid(height: 16, width: 64, window: 32, overlap: 0.5) }
        #expect(throws: PIVError.self) { try Grid(height: 64, width: 64, window: 32, overlap: 1) }
        #expect(throws: PIVError.self) { try Grid(height: 64, width: 64, window: 32, overlap: -0.1) }
    }
}

@Suite struct VectorFieldTests {
    @Test func sampleInterpolatesAndSkipsNaN() throws {
        var f = VectorField(grid: try Grid(height: 64, width: 64, window: 32, overlap: 0)) // centres 15.5, 47.5
        f.dx = [0, 2, 0, 2]
        f.dy = [1, 1, 1, 1]
        #expect(f.sample(x: 31.5, y: 20) == (1, 1))
        #expect(f.sample(x: -100, y: 15.5) == (0, 1))
        f.dx[0] = .nan
        #expect(f.sample(x: 31.5, y: 15.5) == (2, 1))
        #expect(f.sample(x: 15.5, y: 15.5) == (0, 0))
    }
}

@Suite struct ScaleTests {
    func fov(_ axis: FOVAxis, _ d: Double?) -> ScaleSpec {
        ScaleSpec(fovDegrees: 84, fovAxis: axis, sheetDistanceMetres: d)
    }

    @Test func c0141WorkingEstimate() {
        let s = fov(.diagonal, 0.3).resolve(width: 1920, height: 1080)!
        #expect(abs(s.metresPerPixel * 1e3 - 0.245_240_7) < 1e-6)
        #expect(s.method == "pinhole estimate: 84° diagonal fov at 0.3 m over 2202.9 px")
    }

    @Test func axesMatchHandComputation() {
        let t = tan(42.0 * .pi / 180)
        let h = fov(.horizontal, 1).resolve(width: 1920, height: 1080)!
        #expect(abs(h.metresPerPixel - 2 * t / 1920) < 1e-12)
        let v = fov(.vertical, 1).resolve(width: 1920, height: 1080)!
        #expect(abs(v.metresPerPixel - 2 * t / 1080) < 1e-12)
    }

    @Test func pxPerMMOverridesAndFOVAloneIsNotEnough() {
        var s = fov(.diagonal, 0.3)
        s.pxPerMM = 4
        #expect(s.resolve(width: 1920, height: 1080)!.metresPerPixel == 0.25e-3)
        #expect(fov(.diagonal, nil).resolve(width: 1920, height: 1080) == nil)
        #expect(ScaleSpec().resolve(width: 1920, height: 1080) == nil)
    }
}

@Suite struct CaptureRateTests {
    @Test func sonyProbeReadsCaptureRateFromHeaderFixture() throws {
        let url = try #require(Bundle.module.url(
            forResource: "sony_rx100m5a_1000p_header", withExtension: "mp4", subdirectory: "Fixtures"))
        let r = try #require(try SonyNRTProbe().probe(url))
        #expect(r.rate == (try FrameRate(1000)))
        #expect(r.raw == "1000p")
        #expect(r.camera == "Sony DSC-RX100M5A")
    }

    @Test func xmlAttrHandlesOtherRatesAndLookalikeTags() {
        let xml = """
            <NonRealTimeMeta><VideoFrameX captureFps="1p"/>
                <VideoFrame videoCodec="AVC" xcaptureFps="2p" captureFps="240p" formatFps="30p"/></NonRealTimeMeta>
            """
        #expect(xmlAttr(xml, tag: "VideoFrame", attr: "captureFps") == "240p")
        #expect(xmlAttr(xml, tag: "VideoFrame", attr: "formatFps") == "30p")
        #expect(xmlAttr(xml, tag: "Device", attr: "modelName") == nil)
    }

    @Test func nonSonyFileYieldsNone() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("pivkit-\(UUID()).mov")
        defer { try? FileManager.default.removeItem(at: url) }
        var bytes: [UInt8] = [0, 0, 0, 16] + Array("ftypqt  ".utf8) + [0, 0, 0, 0]
        bytes += [0, 0, 0, 1] + Array("mdat".utf8) + [0, 0, 0, 0, 0, 0, 0, 16]
        try Data(bytes).write(to: url)
        #expect(try SonyNRTProbe().probe(url) == nil)
    }

    @Test func resolutionOrderAndWarnings() throws {
        let k = try FrameRate(1000), container = try FrameRate(50), o = try FrameRate(240)
        let cam = CameraRate(rate: k, source: "Sony NonRealTimeMeta", raw: "1000p")
        var r = CaptureRate.resolve(override: nil, camera: cam, container: container)
        #expect(r.rate == k && r.source == .cameraMetadata && r.warning == nil)
        r = CaptureRate.resolve(override: nil, camera: nil, container: container)
        #expect(r.rate.fps == 50 && r.source == .container && r.warning != nil)
        r = CaptureRate.resolve(override: o, camera: cam, container: container)
        #expect(r.rate == o && r.source == .override && r.warning != nil)
        #expect(CaptureRate.resolve(override: k, camera: cam, container: container).warning == nil)
    }
}
