import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import Metal

/// Quiver rendering parameters. Keep them fixed across a run so arrow lengths and colours are
/// comparable between frames. Port of `piv_io::render::QuiverStyle`.
public struct QuiverStyle: Hashable, Sendable {
    /// Arrow length in image pixels per pixel of displacement.
    public var arrowScale: Float
    /// Displacement magnitude (px) at the top of the colour map.
    public var colorMax: Float
    /// Multiplier applied to the background image intensity.
    public var gain: Float
    /// Draw every n-th vector in each direction.
    public var every: Int = 1

    public init(arrowScale: Float, colorMax: Float, gain: Float, every: Int = 1) {
        self.arrowScale = arrowScale
        self.colorMax = colorMax
        self.gain = gain
        self.every = every
    }

    /// Scales arrows so a typical moving vector (75th percentile) spans ~60 % of the grid
    /// spacing, colours up to the 95th percentile, and stretches the background so its 99th
    /// percentile is near white. As `QuiverStyle::auto`.
    public static func auto(raw: [Float], field: VectorField) -> QuiverStyle {
        var mags = zip(zip(field.dx, field.dy), field.flags)
            .filter { $0.1.isValid }
            .map { hypotf($0.0.0, $0.0.1) }
            .filter(\.isFinite)
        let m75 = percentile(&mags, 0.75).flatMap { $0 > 0.1 ? $0 : nil } ?? 1
        let m95 = percentile(&mags, 0.95).flatMap { $0 > 0.1 ? $0 : nil } ?? 1
        var px = Swift.stride(from: 0, to: raw.count, by: 7).map { raw[$0] }
        let p99 = percentile(&px, 0.99).flatMap { $0 > 1e-3 ? $0 : nil } ?? 1
        return QuiverStyle(arrowScale: 0.6 * Float(field.grid.step) / m75, colorMax: m95, gain: min(max(0.9 / p99, 1), 20))
    }

    private static func percentile(_ v: inout [Float], _ q: Float) -> Float? {
        guard !v.isEmpty else { return nil }
        v.sort { totalOrderLess($0, $1) }
        return v[Int((Float(v.count - 1) * q).rounded())]
    }
}

/// Layout matches the Metal `QuiverParams`.
struct QuiverParams {
    var width, height, targetW, targetH: UInt32
    var gain, arrowScale, colorMax, halfWidth: Float
    var nx, every: UInt32
}

/// Layout matches the Metal `QuiverVector`.
struct QuiverVector {
    var center: SIMD2<Float>
    var d: SIMD2<Float>
    var flags: UInt32
    var pad: UInt32 = 0
}

/// Draws quiver frames on the GPU into BGRA textures or pixel buffers.
public final class QuiverRenderer: @unchecked Sendable {
    let ctx: MetalContext
    private let background: MTLRenderPipelineState
    private let strokes: MTLRenderPipelineState
    private let textureCache: CVMetalTextureCache

    public init(context: MetalContext) throws {
        ctx = context
        func pipeline(_ v: String, _ f: String, blend: Bool) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = context.library.makeFunction(name: v)
            d.fragmentFunction = context.library.makeFunction(name: f)
            let c = d.colorAttachments[0]!
            c.pixelFormat = .bgra8Unorm
            if blend {
                c.isBlendingEnabled = true
                c.sourceRGBBlendFactor = .sourceAlpha
                c.destinationRGBBlendFactor = .oneMinusSourceAlpha
                c.sourceAlphaBlendFactor = .one
                c.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            return try context.device.makeRenderPipelineState(descriptor: d)
        }
        background = try pipeline("quiver_background_vertex", "quiver_background_fragment", blend: false)
        strokes = try pipeline("quiver_stroke_vertex", "quiver_stroke_fragment", blend: true)
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, context.device, nil, &cache) == kCVReturnSuccess, let cache else {
            throw PIVError.metal("cannot create a texture cache")
        }
        textureCache = cache
    }

    /// Encodes one quiver frame of `field` over `raw` (ROI-sized intensities) into `target`.
    func encode(_ cb: MTLCommandBuffer, target: MTLTexture, raw: MTLBuffer, width: Int, height: Int,
                field: VectorField, style: QuiverStyle) throws {
        let g = field.grid
        var vectors = [QuiverVector]()
        vectors.reserveCapacity(g.count)
        for iy in 0..<g.ny {
            for ix in 0..<g.nx {
                let k = iy * g.nx + ix
                vectors.append(QuiverVector(center: SIMD2(g.centerX(ix), g.centerY(iy)), d: SIMD2(field.dx[k], field.dy[k]),
                                            flags: UInt32(field.flags[k].rawValue)))
            }
        }
        var p = QuiverParams(
            width: UInt32(width), height: UInt32(height), targetW: UInt32(target.width), targetH: UInt32(target.height),
            gain: style.gain, arrowScale: style.arrowScale, colorMax: style.colorMax,
            halfWidth: width >= 1280 ? 0.85 : 0.5, nx: UInt32(g.nx), every: UInt32(max(style.every, 1)))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { throw PIVError.metal("no render encoder") }
        enc.setRenderPipelineState(background)
        enc.setFragmentBuffer(raw, offset: 0, index: 0)
        enc.setFragmentBytes(&p, length: MemoryLayout<QuiverParams>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        if !vectors.isEmpty {
            enc.setRenderPipelineState(strokes)
            enc.setVertexBuffer(try ctx.makeBuffer(vectors), offset: 0, index: 0)
            enc.setVertexBytes(&p, length: MemoryLayout<QuiverParams>.stride, index: 1)
            enc.setFragmentBytes(&p, length: MemoryLayout<QuiverParams>.stride, index: 0)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: vectors.count * 3)
        }
        enc.endEncoding()
    }

    /// Renders one frame into a pixel buffer (BGRA, Metal-compatible) and waits for it.
    func render(into pixelBuffer: CVPixelBuffer, raw: MTLBuffer, width: Int, height: Int,
                field: VectorField, style: QuiverStyle) async throws {
        var tex: CVMetalTexture?
        let (w, h) = (CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer))
        guard CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pixelBuffer, nil, .bgra8Unorm, w, h, 0, &tex) == kCVReturnSuccess,
              let tex, let target = CVMetalTextureGetTexture(tex),
              let cb = ctx.queue.makeCommandBuffer()
        else { throw PIVError.metal("cannot render into the movie frame") }
        try encode(cb, target: target, raw: raw, width: width, height: height, field: field, style: style)
        try await ctx.run(cb)
        withExtendedLifetime(tex) {}
    }

    /// Renders one frame to an image (previews, stills).
    public func image(raw: MTLBuffer, width: Int, height: Int, field: VectorField, style: QuiverStyle) async throws -> CGImage {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .shared
        guard let target = ctx.device.makeTexture(descriptor: d), let cb = ctx.queue.makeCommandBuffer() else {
            throw PIVError.metal("cannot create a preview texture")
        }
        try encode(cb, target: target, raw: raw, width: width, height: height, field: field, style: style)
        try await ctx.run(cb)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        target.getBytes(&bytes, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw PIVError.io("cannot make a preview image") }
        return img
    }
}

/// Writes quiver frames to a movie with AVAssetWriter.
public final class MovieWriter: @unchecked Sendable {
    public enum Codec: String, Sendable, CaseIterable { case h264, hevc }

    public let url: URL
    public let width: Int, height: Int
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let fps: Double
    private var frames = 0

    /// `width × height` is the image size; the movie is padded to even dimensions.
    public init(url: URL, width: Int, height: Int, fps: Double, codec: Codec = .h264) throws {
        self.url = url
        self.fps = fps
        self.width = (width + 1) / 2 * 2
        self.height = (height + 1) / 2 * 2
        try? FileManager.default.removeItem(at: url)
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: codec == .hevc ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
            AVVideoWidthKey: self.width,
            AVVideoHeightKey: self.height,
            // ~0.25 bit per pixel: thin coloured arrows stay sharp at a size comparable to
            // the Rust tool's x264 CRF 18 output.
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: Int(Double(self.width * self.height) * fps * 0.25)],
        ])
        input.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: self.width,
            kCVPixelBufferHeightKey as String: self.height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ])
        writer.add(input)
        guard writer.startWriting() else {
            throw PIVError.io("cannot write \(url.lastPathComponent): \(writer.error?.localizedDescription ?? "unknown error")")
        }
        writer.startSession(atSourceTime: .zero)
    }

    /// A pixel buffer to render the next frame into.
    func makePixelBuffer() throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        guard let pool = adaptor.pixelBufferPool,
              CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess, let pb
        else { throw PIVError.io("movie writer has no pixel buffers: \(writer.error?.localizedDescription ?? "")") }
        return pb
    }

    /// Appends the next frame (in order).
    func append(_ pb: CVPixelBuffer) async throws {
        while !input.isReadyForMoreMediaData {
            if writer.status == .failed { break }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        let t = CMTime(value: CMTimeValue(frames) * 1000, timescale: CMTimeScale((fps * 1000).rounded()))
        guard adaptor.append(pb, withPresentationTime: t) else {
            throw PIVError.io("writing frame \(frames): \(writer.error?.localizedDescription ?? "unknown error")")
        }
        frames += 1
    }

    public func finish() async throws {
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed {
            throw PIVError.io("finishing \(url.lastPathComponent): \(writer.error?.localizedDescription ?? "unknown error")")
        }
    }

    public func cancel() { writer.cancelWriting() }
}
