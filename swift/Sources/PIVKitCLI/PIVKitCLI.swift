import ArgumentParser
import Foundation
import PIVKit

@main
struct PIVKitCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pivkit",
        abstract: "Planar image velocimetry on the GPU (PIVKit).",
        subcommands: [Bench.self]
    )
}

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Time the GPU correlation kernels on a synthetic 1920×1080 pair.")

    @Option(help: "Timed repeats per measurement (median reported).")
    var repeats = 9

    func run() async throws {
        let ctx = try MetalContext()
        print("device: \(ctx.device.name), max threadgroup memory \(ctx.device.maxThreadgroupMemoryLength) B")
        print("")
        print("window  kernel                                   GPU ms")
        for r in try await CorrelationBenchmark.correlation(context: ctx, repeats: repeats) {
            print(String(format: "%4d px  %-40@ %7.3f", r.windows, r.label as NSString, r.milliseconds))
        }
        print("")
        for r in try await CorrelationBenchmark.pair(context: ctx, repeats: repeats) {
            print(String(format: "%-48@ %7.3f ms  (%.0f pairs/s)", r.label as NSString, r.milliseconds, 1000 / r.milliseconds))
        }
    }
}
