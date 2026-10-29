import ArgumentParser
import PIVKit

@main
struct PIVKitCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pivkit",
        abstract: "Planar image velocimetry on the GPU (PIVKit).",
        subcommands: []
    )
}
