import Foundation
import Metal

/// The Metal device, command queue and compiled kernels shared by PIVKit's GPU stages.
///
/// The kernels ship as `.metal` source in the package's resources and are compiled at runtime
/// with fast math off, so results track the Rust reference as closely as f32 allows. Device,
/// queue, library and pipeline states are thread-safe; this class adds a lock around the
/// pipeline cache only.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    /// Kernel sources, compiled as one unit in this order (shared definitions first).
    static let sourceOrder = ["Common", "Correlate"]

    public convenience init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw PIVError.metal("no Metal device") }
        try self.init(device: device)
    }

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw PIVError.metal("cannot create a command queue") }
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        do {
            library = try device.makeLibrary(source: try Self.kernelSource(), options: options)
        } catch let e as PIVError {
            throw e
        } catch {
            throw PIVError.metal("compiling kernels: \(error)")
        }
    }

    static func kernelSource() throws -> String {
        guard let dir = Bundle.module.url(forResource: "Metal", withExtension: nil) else {
            throw PIVError.metal("kernel sources missing from the PIVKit bundle")
        }
        return try sourceOrder.map { name in
            try String(contentsOf: dir.appendingPathComponent("\(name).metal"), encoding: .utf8)
        }.joined(separator: "\n")
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let f = library.makeFunction(name: name) else { throw PIVError.metal("no kernel \(name)") }
        let p = try device.makeComputePipelineState(function: f)
        pipelines[name] = p
        return p
    }

    func makeBuffer<T>(_ values: [T]) throws -> MTLBuffer {
        let length = max(values.count * MemoryLayout<T>.stride, 16)
        let b = values.withUnsafeBytes { raw in
            raw.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: raw.count, options: .storageModeShared) }
        } ?? device.makeBuffer(length: length, options: .storageModeShared)
        guard let b else { throw PIVError.metal("cannot allocate \(length) bytes") }
        return b
    }

    func makeBuffer(length: Int) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(length, 16), options: .storageModeShared) else {
            throw PIVError.metal("cannot allocate \(length) bytes")
        }
        return b
    }

    /// Commits `cb` and waits for it without blocking a thread. Throws if the GPU reported an
    /// error.
    func run(_ cb: MTLCommandBuffer) async throws {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            cb.addCompletedHandler { _ in c.resume() }
            cb.commit()
        }
        if let error = cb.error { throw PIVError.metal("command buffer failed: \(error)") }
    }
}

/// A frame on the GPU, ready for correlation: the conditioned image, plus the
/// background-subtracted raw intensity used for the window signal mask. Both are dense
/// row-major `Float` buffers of `width × height`.
public struct GPUFrame: @unchecked Sendable {
    public let image: MTLBuffer
    public let stats: MTLBuffer
    public let width: Int
    public let height: Int

    public init(image: MTLBuffer, stats: MTLBuffer, width: Int, height: Int) {
        self.image = image
        self.stats = stats
        self.width = width
        self.height = height
    }
}

extension MetalContext {
    /// A frame from CPU arrays (tests and tools; the video pipeline prepares frames on the GPU).
    public func makeFrame(image: [Float], stats: [Float], width: Int, height: Int) throws -> GPUFrame {
        precondition(image.count == width * height && stats.count == width * height)
        return GPUFrame(image: try makeBuffer(image), stats: try makeBuffer(stats), width: width, height: height)
    }
}
