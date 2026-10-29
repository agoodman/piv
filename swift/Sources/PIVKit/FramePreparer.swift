import CoreVideo
import Foundation
import Metal

/// Image conditioning applied before correlation. Port of `piv_core::PreprocessConfig`; the
/// defaults match.
public struct PreprocessConfig: Hashable, Sendable, Codable {
    /// Gaussian smoothing sigma in pixels (0 disables). Suppresses sensor noise.
    public var smoothSigma: Float = 1
    /// Half-size of the box used for local mean/std normalization (0 disables).
    public var normalizeRadius = 4
    /// Added to the local std before dividing, so dark noisy regions are not amplified.
    public var normalizeEps: Float = 0.02
    public init() {}
}

/// How a static background image is estimated from frames sampled across the whole clip.
public struct BackgroundSpec: Hashable, Sendable, Codable {
    public enum Mode: String, Hashable, Sendable, Codable, CaseIterable { case min, mean }
    public var mode: Mode
    public var frames: Int
    public init(mode: Mode = .min, frames: Int = 40) {
        self.mode = mode
        self.frames = frames
    }

    /// Parses `none`, `min:N` or `mean:N` (the Rust `--background` syntax).
    public static func parse(_ s: String) throws(PIVError) -> BackgroundSpec? {
        if s == "none" { return nil }
        let parts = s.split(separator: ":")
        guard parts.count == 2, let mode = Mode(rawValue: String(parts[0])), let n = Int(parts[1]), n >= 1 else {
            throw .config("background \(s.debugDescription): expected none, min:N or mean:N")
        }
        return BackgroundSpec(mode: mode, frames: n)
    }

    public var description: String { "\(mode.rawValue):\(frames)" }
}

/// A frame on the GPU after preparation, plus the uploaded intensity for rendering. Its
/// buffers go back to the preparer's pool when the last copy is released.
public struct PreparedFrame: @unchecked Sendable {
    public let index: Int
    public let gpu: GPUFrame
    /// The uploaded (ROI) intensity before background subtraction.
    public let raw: MTLBuffer
    let storage: [PooledBuffer]
}

/// Recycles equally sized GPU buffers, so frames don't each allocate (and the system doesn't
/// zero-fill) fresh memory.
final class BufferPool: @unchecked Sendable {
    private let ctx: MetalContext
    private let length: Int
    private var free: [MTLBuffer] = []
    private let lock = NSLock()

    init(context: MetalContext, length: Int) {
        ctx = context
        self.length = length
    }

    func take() throws -> PooledBuffer {
        lock.lock()
        let b = free.popLast()
        lock.unlock()
        return PooledBuffer(buffer: try b ?? ctx.makeBuffer(length: length), pool: self)
    }

    fileprivate func giveBack(_ b: MTLBuffer) {
        lock.lock()
        free.append(b)
        lock.unlock()
    }
}

/// A buffer on loan from a ``BufferPool``; returned when this object is released.
final class PooledBuffer: @unchecked Sendable {
    let buffer: MTLBuffer
    private let pool: BufferPool
    init(buffer: MTLBuffer, pool: BufferPool) {
        self.buffer = buffer
        self.pool = pool
    }
    deinit { pool.giveBack(buffer) }
}

/// Uploads decoded frames to the GPU and prepares them for correlation (port of
/// `PreparedFrame::new`): optional background subtraction (clamped at 0), Gaussian blur,
/// local contrast normalization. The background-subtracted intensity is kept for the window
/// signal mask.
public final class FramePreparer: @unchecked Sendable {
    let ctx: MetalContext
    public let config: PreprocessConfig
    public let roi: ROI
    public let channel: Channel
    private let textureCache: CVMetalTextureCache
    private let blurWeights: MTLBuffer?
    private let blurRadius: Int
    private(set) public var background: MTLBuffer?
    private let pool: BufferPool
    // Intermediates, reused: preparations run one at a time (each awaits its GPU work).
    private var scratchTmp: MTLBuffer?
    private var scratchBlur: MTLBuffer?
    private var scratchSums: MTLBuffer?

    public init(context: MetalContext, config: PreprocessConfig, roi: ROI, channel: Channel) throws {
        ctx = context
        self.config = config
        self.roi = roi
        self.channel = channel
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, context.device, nil, &cache) == kCVReturnSuccess, let cache else {
            throw PIVError.metal("cannot create a texture cache")
        }
        textureCache = cache
        pool = BufferPool(context: context, length: roi.width * roi.height * MemoryLayout<Float>.stride)
        if config.smoothSigma > 0 {
            let (w, r) = Self.gaussianKernel(sigma: config.smoothSigma)
            blurWeights = try context.makeBuffer(w)
            blurRadius = r
        } else {
            blurWeights = nil
            blurRadius = 0
        }
    }

    /// Normalized Gaussian weights as `piv_core::preprocess::gaussian_blur` computes them (f32).
    static func gaussianKernel(sigma: Float) -> ([Float], Int) {
        let r = Int(max((3 * sigma).rounded(.up), 1))
        var k = (-r...r).map { i in exp(-Float(i * i) / (2 * sigma * sigma)) }
        let total = k.reduce(Float(0), +)
        k = k.map { $0 / total }
        return (k, r)
    }

    private var params: ImageParams {
        ImageParams(width: UInt32(roi.width), height: UInt32(roi.height), srcX: UInt32(roi.x), srcY: UInt32(roi.y),
                    radius: 0, eps: 0, channel: channel == .green ? 1 : 0, flag: 0)
    }

    private func floatBuffer() throws -> MTLBuffer {
        try ctx.makeBuffer(length: roi.width * roi.height * MemoryLayout<Float>.stride)
    }

    private func scratch(_ b: inout MTLBuffer?, bytesPerPixel: Int = MemoryLayout<Float>.stride) throws -> MTLBuffer {
        if let b { return b }
        let n = try ctx.makeBuffer(length: roi.width * roi.height * bytesPerPixel)
        b = n
        return n
    }

    private func texture(_ pb: CVPixelBuffer) throws -> CVMetalTexture {
        let planar = CVPixelBufferIsPlanar(pb)
        let w = planar ? CVPixelBufferGetWidthOfPlane(pb, 0) : CVPixelBufferGetWidth(pb)
        let h = planar ? CVPixelBufferGetHeightOfPlane(pb, 0) : CVPixelBufferGetHeight(pb)
        guard roi.x + roi.width <= w, roi.y + roi.height <= h else {
            throw PIVError.config("ROI \(roi.x),\(roi.y),\(roi.width),\(roi.height) exceeds the \(w)×\(h) frame")
        }
        let format: MTLPixelFormat = switch CVPixelBufferGetPixelFormatType(pb) {
        case kCVPixelFormatType_32BGRA: .bgra8Unorm
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr10BiPlanarFullRange: .r16Unorm
        default: .r8Unorm
        }
        var tex: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pb, nil, format, w, h, 0, &tex)
        guard status == kCVReturnSuccess, let tex else { throw PIVError.metal("cannot map the decoded frame (\(status))") }
        return tex
    }

    private func dispatch2D(_ enc: MTLComputeCommandEncoder) {
        enc.dispatchThreads(MTLSize(width: roi.width, height: roi.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder) {
        enc.dispatchThreads(MTLSize(width: roi.width * roi.height, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Uploads `frame` (ROI) into `raw`, and the background-subtracted base into `base`.
    private func encodeUpload(_ cb: MTLCommandBuffer, _ tex: CVMetalTexture, raw: MTLBuffer, base: MTLBuffer) throws {
        guard let t = CVMetalTextureGetTexture(tex), let enc = cb.makeComputeCommandEncoder() else {
            throw PIVError.metal("cannot encode the frame upload")
        }
        var p = params
        p.flag = background == nil ? 0 : 1
        enc.setComputePipelineState(try ctx.pipeline("upload_frame"))
        enc.setTexture(t, index: 0)
        enc.setBytes(&p, length: MemoryLayout<ImageParams>.stride, index: 0)
        enc.setBuffer(raw, offset: 0, index: 1)
        enc.setBuffer(base, offset: 0, index: 2)
        enc.setBuffer(background ?? base, offset: 0, index: 3)
        dispatch2D(enc)
        enc.endEncoding()
    }

    /// Uploads and prepares one decoded frame.
    /// Encodes and commits the preparation of one decoded frame, without waiting for it.
    /// Work committed later to the same queue (the correlation of pairs using this frame) runs
    /// after it, so callers never need to wait. The decoded picture stays alive until the GPU
    /// is done with it; a GPU failure is reported by the next call (or ``checkFailure()``).
    public func prepare(_ frame: DecodedFrame) throws -> PreparedFrame {
        try checkFailure()
        let tex = try texture(frame.pixelBuffer)
        guard let cb = ctx.queue.makeCommandBuffer() else { throw PIVError.metal("no command buffer") }
        var storage = [try pool.take()]
        let base = storage[0].buffer
        var raw = base
        if background != nil {
            storage.append(try pool.take())
            raw = storage[1].buffer
        }
        try encodeUpload(cb, tex, raw: raw, base: base)
        let image = try encodeConditioning(cb, base: base, storage: &storage)
        let keep = Retained((tex, frame))
        cb.addCompletedHandler { [weak self] cb in
            withExtendedLifetime(keep) {}
            if let e = cb.error { self?.record(PIVError.metal("frame preparation failed: \(e)")) }
        }
        cb.commit()
        return PreparedFrame(index: frame.index, gpu: GPUFrame(image: image, stats: base, width: roi.width, height: roi.height),
                             raw: raw, storage: storage)
    }

    /// Uploads a frame's intensity (ROI, before background subtraction) without conditioning,
    /// for drawing. Commits without waiting, like ``prepare(_:)``.
    public func uploadRaw(_ frame: DecodedFrame) throws -> PreparedFrame {
        try checkFailure()
        let tex = try texture(frame.pixelBuffer)
        guard let cb = ctx.queue.makeCommandBuffer() else { throw PIVError.metal("no command buffer") }
        let storage = [try pool.take()]
        let raw = storage[0].buffer
        let saved = background
        background = nil
        defer { background = saved }
        try encodeUpload(cb, tex, raw: raw, base: raw)
        let keep = Retained((tex, frame))
        cb.addCompletedHandler { [weak self] cb in
            withExtendedLifetime(keep) {}
            if let e = cb.error { self?.record(PIVError.metal("frame upload failed: \(e)")) }
        }
        cb.commit()
        return PreparedFrame(index: frame.index, gpu: GPUFrame(image: raw, stats: raw, width: roi.width, height: roi.height),
                             raw: raw, storage: storage)
    }

    private var failure: Error?
    private let failureLock = NSLock()

    private func record(_ e: Error) {
        failureLock.withLock { if failure == nil { failure = e } }
    }

    /// Throws the first GPU failure of an earlier ``prepare(_:)``, if any.
    public func checkFailure() throws {
        if let e = failureLock.withLock({ failure }) { throw e }
    }

    /// Blur and local normalization of `base`; returns the conditioned image (or `base` when
    /// both are off). The output buffer comes from the pool and is appended to `storage`.
    private func encodeConditioning(_ cb: MTLCommandBuffer, base: MTLBuffer, storage: inout [PooledBuffer]) throws -> MTLBuffer {
        let blur = blurWeights != nil, normalize = config.normalizeRadius > 0
        guard blur || normalize else { return base }
        let out = try pool.take()
        storage.append(out)
        guard let enc = cb.makeComputeCommandEncoder() else { throw PIVError.metal("no encoder") }
        defer { enc.endEncoding() }
        var p = params
        var image = base
        if let weights = blurWeights {
            let tmp = try scratch(&scratchTmp)
            let blurred = normalize ? try scratch(&scratchBlur) : out.buffer
            p.radius = UInt32(blurRadius)
            enc.setComputePipelineState(try ctx.pipeline("blur_rows"))
            enc.setBuffer(image, offset: 0, index: 0)
            enc.setBuffer(tmp, offset: 0, index: 1)
            enc.setBytes(&p, length: MemoryLayout<ImageParams>.stride, index: 2)
            enc.setBuffer(weights, offset: 0, index: 3)
            dispatch2D(enc)
            enc.setComputePipelineState(try ctx.pipeline("blur_cols"))
            enc.setBuffer(tmp, offset: 0, index: 0)
            enc.setBuffer(blurred, offset: 0, index: 1)
            dispatch2D(enc)
            image = blurred
        }
        if normalize {
            let sums = try scratch(&scratchSums, bytesPerPixel: MemoryLayout<SIMD2<Float>>.stride)
            p.radius = UInt32(config.normalizeRadius)
            p.eps = config.normalizeEps
            enc.setComputePipelineState(try ctx.pipeline("box_rows"))
            enc.setBuffer(image, offset: 0, index: 0)
            enc.setBuffer(sums, offset: 0, index: 1)
            enc.setBytes(&p, length: MemoryLayout<ImageParams>.stride, index: 2)
            dispatch2D(enc)
            enc.setComputePipelineState(try ctx.pipeline("normalize_cols"))
            enc.setBuffer(image, offset: 0, index: 0)
            enc.setBuffer(sums, offset: 0, index: 1)
            enc.setBuffer(out.buffer, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<ImageParams>.stride, index: 3)
            dispatch2D(enc)
        }
        return out.buffer
    }

    /// Prepares a frame given as CPU intensities (ROI-sized, row-major): tests and tools.
    public func prepare(raw values: [Float], index: Int = 0) async throws -> PreparedFrame {
        precondition(values.count == roi.width * roi.height)
        let raw = try ctx.makeBuffer(values)
        let base: MTLBuffer
        if let bg = background {
            base = try ctx.makeBuffer(zip(values, bg.floats(values.count)).map { max($0 - $1, 0) })
        } else {
            base = raw
        }
        guard let cb = ctx.queue.makeCommandBuffer() else { throw PIVError.metal("no command buffer") }
        var storage: [PooledBuffer] = []
        let image = try encodeConditioning(cb, base: base, storage: &storage)
        try await ctx.run(cb)
        return PreparedFrame(index: index, gpu: GPUFrame(image: image, stats: base, width: roi.width, height: roi.height),
                             raw: raw, storage: storage)
    }

    /// Estimates the static background from `spec.frames` frames sampled evenly across the
    /// whole clip (frame k·(total − 1)/(n − 1)), as `piv analyze --background` does. Frames
    /// prepared afterwards have it subtracted.
    public func estimateBackground(_ spec: BackgroundSpec, info: MediaInfo, progress: ((Int, Int) -> Void)? = nil) async throws {
        background = nil
        let total = info.frameCount
        let n = min(spec.frames, total)
        let acc = try floatBuffer()
        for k in 0..<n {
            let idx = n > 1 ? k * (total - 1) / (n - 1) : 0
            let frame = try await VideoDecoder.frame(idx, info: info, channel: channel)
            let tex = try texture(frame.pixelBuffer)
            guard let cb = ctx.queue.makeCommandBuffer() else { throw PIVError.metal("no command buffer") }
            let raw = try floatBuffer()
            try encodeUpload(cb, tex, raw: raw, base: raw)
            guard let enc = cb.makeComputeCommandEncoder() else { throw PIVError.metal("no encoder") }
            var p = params
            p.flag = k == 0 ? 2 : spec.mode == .min ? 0 : 1
            enc.setComputePipelineState(try ctx.pipeline("accumulate_background"))
            enc.setBuffer(acc, offset: 0, index: 0)
            enc.setBuffer(raw, offset: 0, index: 1)
            enc.setBytes(&p, length: MemoryLayout<ImageParams>.stride, index: 2)
            dispatch1D(enc)
            if spec.mode == .mean && k == n - 1 {
                var count = Float(n)
                enc.setComputePipelineState(try ctx.pipeline("scale_image"))
                enc.setBuffer(acc, offset: 0, index: 0)
                enc.setBytes(&p, length: MemoryLayout<ImageParams>.stride, index: 1)
                enc.setBytes(&count, length: 4, index: 2)
                dispatch1D(enc)
            }
            enc.endEncoding()
            try await ctx.run(cb)
            withExtendedLifetime(tex) {}
            progress?(k + 1, n)
        }
        background = acc
    }

    /// Uses a given background image (ROI-sized, row-major), e.g. from tests.
    public func setBackground(_ values: [Float]?) throws {
        background = try values.map { try ctx.makeBuffer($0) }
    }
}

/// Layout matches the Metal `ImageParams`.
struct ImageParams {
    var width: UInt32
    var height: UInt32
    var srcX: UInt32
    var srcY: UInt32
    var radius: UInt32
    var eps: Float
    var channel: UInt32
    var flag: UInt32
}

extension MTLBuffer {
    /// Copies the first `count` floats out of a shared buffer.
    func floats(_ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: contents().bindMemory(to: Float.self, capacity: count), count: count))
    }
}

/// Keeps non-Sendable objects alive inside a GPU completion handler.
private final class Retained<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}
