import Foundation

/// Writes `.npy` files (NumPy format 1.0), for comparing runs with the Rust tool's output in
/// Python. Little-endian, C order.
enum Npy {
    enum DType: String {
        case f4 = "<f4", f8 = "<f8", u1 = "|u1", u2 = "<u2", u8 = "<u8"
    }

    static func data(shape: [Int], dtype: DType, body: Data) -> Data {
        let shapeText = shape.count == 1 ? "(\(shape[0]),)" : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        var header = "{'descr': '\(dtype.rawValue)', 'fortran_order': False, 'shape': \(shapeText), }"
        // Magic (6) + version (2) + length (2) + header, padded with spaces to a multiple of 64
        // and ending in a newline.
        let unpadded = 10 + header.utf8.count + 1
        header += String(repeating: " ", count: (64 - unpadded % 64) % 64) + "\n"
        var d = Data([0x93] + Array("NUMPY".utf8) + [1, 0])
        let n = UInt16(header.utf8.count)
        d.append(contentsOf: [UInt8(n & 0xff), UInt8(n >> 8)])
        d.append(contentsOf: Array(header.utf8))
        d.append(body)
        return d
    }

    static func write<T>(_ values: [T], shape: [Int], dtype: DType, to url: URL) throws {
        precondition(shape.reduce(1, *) == values.count, "shape \(shape) for \(values.count) values")
        let body = values.withUnsafeBytes { Data($0) }
        try data(shape: shape, dtype: dtype, body: body).write(to: url)
    }
}

extension RunResult {
    /// Writes the run as a directory of `.npy` files plus `run.json`, in the layout of the
    /// Rust tool's HDF5 output (dataset `fields/dx` → `dx.npy`, `stats/ke_sum` →
    /// `stats_ke_sum.npy`). With smoothing, the top level holds the smoothed fields (as
    /// `piv smooth` writes them) and `raw/` the validated raw fields.
    public func writeNpy(to dir: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let (nt, ny, nx) = (pairs.count, grid.ny, grid.nx)
        let shape = [nt, ny, nx]
        let invDt = Float(1 / dt)

        func writeFields(_ fields: [VectorField], stats: [FlowStats], in d: URL, peakRatio: Bool) throws {
            try fm.createDirectory(at: d, withIntermediateDirectories: true)
            let dx = fields.flatMap(\.dx), dy = fields.flatMap(\.dy)
            let u = dx.map { $0 * invDt }, v = dy.map { $0 * invDt }
            try Npy.write(dx, shape: shape, dtype: .f4, to: d.appendingPathComponent("dx.npy"))
            try Npy.write(dy, shape: shape, dtype: .f4, to: d.appendingPathComponent("dy.npy"))
            try Npy.write(u, shape: shape, dtype: .f4, to: d.appendingPathComponent("u.npy"))
            try Npy.write(v, shape: shape, dtype: .f4, to: d.appendingPathComponent("v.npy"))
            if let k = scale.map({ Float($0.metresPerPixel) }) {
                try Npy.write(u.map { $0 * k }, shape: shape, dtype: .f4, to: d.appendingPathComponent("u_mps.npy"))
                try Npy.write(v.map { $0 * k }, shape: shape, dtype: .f4, to: d.appendingPathComponent("v_mps.npy"))
            }
            if peakRatio {
                try Npy.write(fields.flatMap(\.peakRatio), shape: shape, dtype: .f4, to: d.appendingPathComponent("peak_ratio.npy"))
            }
            try Npy.write(fields.flatMap { $0.flags.map(\.rawValue) }, shape: shape, dtype: .u1, to: d.appendingPathComponent("flags.npy"))
            for (name, column) in [
                ("vorticity_mean", stats.map(\.vorticityMean)), ("vorticity_abs_mean", stats.map(\.vorticityAbsMean)),
                ("ke_mean", stats.map(\.keMean)), ("ke_sum", stats.map(\.keSum)), ("coverage", stats.map(\.coverage)),
            ] {
                try Npy.write(column, shape: [nt], dtype: .f4, to: d.appendingPathComponent("stats_\(name).npy"))
            }
        }

        if let smoothed, let support, let smoothedStats {
            try writeFields(smoothed, stats: smoothedStats, in: dir, peakRatio: false)
            try Npy.write(support.flatMap { $0 }, shape: shape, dtype: .u2, to: dir.appendingPathComponent("support.npy"))
            try writeFields(raw, stats: rawStats, in: dir.appendingPathComponent("raw"), peakRatio: true)
        } else {
            try writeFields(raw, stats: rawStats, in: dir, peakRatio: true)
        }
        // Grid in full-frame pixels (ROI offset added back), and the pairs.
        try Npy.write(grid.xs.map { $0 + Float(roi.x) }, shape: [nx], dtype: .f4, to: dir.appendingPathComponent("grid_x.npy"))
        try Npy.write(grid.ys.map { $0 + Float(roi.y) }, shape: [ny], dtype: .f4, to: dir.appendingPathComponent("grid_y.npy"))
        try Npy.write(pairs.map { UInt64($0.frameA) }, shape: [nt], dtype: .u8, to: dir.appendingPathComponent("frame_a.npy"))
        try Npy.write(pairs.map { UInt64($0.frameB) }, shape: [nt], dtype: .u8, to: dir.appendingPathComponent("frame_b.npy"))
        try Npy.write(pairs.map(\.time), shape: [nt], dtype: .f8, to: dir.appendingPathComponent("t.npy"))

        let json = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: dir.appendingPathComponent("run.json"))
    }

    /// Run parameters, using the Rust `run_json` key names where they exist.
    public var metadata: [String: Any] {
        let r = request
        var m: [String: Any] = [
            "tool_version": "pivkit \(PIVKitVersion.string)",
            "implementation": "pivkit \(PIVKitVersion.string) (Swift + Metal)",
            "source_path": info.url.path,
            "source_size": info.fileSize,
            "frame_width": info.width,
            "frame_height": info.height,
            "channel": r.channel == .luma ? "yplane" : "green",
            "capture_fps": capture.rate.fps,
            "capture_fps_rational": "\(capture.rate.num)/\(capture.rate.den)",
            "capture_fps_source": capture.source.rawValue,
            "capture_fps_detail": capture.detail,
            "container_fps": info.containerRate.fps,
            "start_frame": frames.lowerBound,
            "end_frame": frames.upperBound,
            "stride": r.stride,
            "pair_step": r.pairStep,
            "dt_s": dt,
            "preprocess": [
                "smooth_sigma": r.preprocess.smoothSigma, "normalize_radius": r.preprocess.normalizeRadius,
                "normalize_eps": r.preprocess.normalizeEps,
            ],
            "piv": [
                "windows": r.piv.windows, "overlap": r.piv.overlap, "final_iterations": r.piv.finalIterations,
                "mask": ["min_mean": r.piv.mask.minMean, "min_std": r.piv.mask.minStd],
                "validation": [
                    "ppr_threshold": r.piv.validation.pprThreshold, "median_threshold": r.piv.validation.medianThreshold,
                    "median_eps": r.piv.validation.medianEps, "replace": r.piv.validation.replace,
                ],
            ],
            "correlation_path": r.path.rawValue,
            "elapsed_s": elapsed,
        ]
        if let c = info.cameraRate?.camera { m["camera"] = c }
        if let roi = r.roi { m["roi"] = ["x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height] }
        if let bg = r.background { m["background"] = bg.description }
        if let s = scale { m["scale"] = ["m_per_px": s.metresPerPixel, "method": s.method] }
        if let t = r.sheetThicknessMM { m["sheet_thickness_mm"] = t }
        if let ts = r.temporalSmoothing {
            m["smoothing"] = ["temporal": ts.window, "temporal_kernel": ts.kernel.rawValue, "min_support": ts.resolvedMinSupport]
        }
        return m
    }
}

public enum PIVKitVersion {
    public static let string = "0.1.0"
}
