// swift-tools-version: 6.0
// PIVKit: planar image velocimetry on the GPU with Metal. Sources live under swift/; the
// manifest is at the repository root so the package can be added by git URL.

import PackageDescription

let package = Package(
    name: "PIVKit",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "PIVKit", targets: ["PIVKit"]),
        .executable(name: "pivkit", targets: ["PIVKitCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "PIVKit",
            path: "swift/Sources/PIVKit",
            resources: [.copy("Metal")]
        ),
        .executableTarget(
            name: "PIVKitCLI",
            dependencies: [
                "PIVKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "swift/Sources/PIVKitCLI"
        ),
        .testTarget(
            name: "PIVKitTests",
            dependencies: ["PIVKit"],
            path: "swift/Tests/PIVKitTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
