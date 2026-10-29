import Foundation

public enum PIVError: Error, Equatable, Sendable, LocalizedError {
    case frameRate(String)
    case config(String)
    case imageTooSmall(height: Int, width: Int, window: Int)
    case shapeMismatch(a: Size, b: Size)
    case metal(String)
    case io(String)

    public struct Size: Equatable, Sendable {
        public let height: Int, width: Int
        public init(height: Int, width: Int) { (self.height, self.width) = (height, width) }
    }

    public var errorDescription: String? {
        switch self {
        case .frameRate(let s): "invalid frame rate \(s.debugDescription)"
        case .config(let s): "invalid configuration: \(s)"
        case let .imageTooSmall(h, w, n): "image \(w)×\(h) is smaller than the \(n) px window"
        case let .shapeMismatch(a, b): "frame sizes differ: \(a.width)×\(a.height) vs \(b.width)×\(b.height)"
        case .metal(let s): "Metal: \(s)"
        case .io(let s): s
        }
    }
}
