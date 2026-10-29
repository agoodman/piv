import PIVKit
import SwiftUI

struct RunningView: View {
    @Environment(AppModel.self) private var model
    let status: AppModel.RunStatus

    var body: some View {
        VStack(spacing: 18) {
            Group {
                if let img = status.preview {
                    Image(decorative: img, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    Rectangle().fill(.black).aspectRatio(16 / 9, contentMode: .fit)
                        .overlay { ProgressView().controlSize(.large) }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .frame(maxWidth: 1100)

            VStack(alignment: .leading, spacing: 8) {
                Text(stageLabel).font(.headline)
                if status.total > 0 {
                    ProgressView(value: Double(status.done), total: Double(status.total))
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
                Text(detail)
                    .font(.callout)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: 640)

            Button("Cancel", role: .cancel) { model.cancel() }
                .keyboardShortcut(.cancelAction)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var stageLabel: String {
        switch status.stage {
        case .background: "Estimating the background…"
        case .analysing: "Analysing frame pairs…"
        case .smoothing: "Smoothing…"
        case .rendering: "Rendering the quiver movie…"
        }
    }

    private var detail: String {
        var parts: [String] = []
        if status.total > 0 {
            let unit = switch status.stage {
            case .background: "frames"
            case .rendering: "movie frames"
            default: "pairs"
            }
            parts.append("\(status.done) of \(status.total) \(unit)")
        }
        if let r = status.pairsPerSecond { parts.append(String(format: "%.0f pairs/s", r)) }
        if let eta = status.eta, eta.isFinite { parts.append(eta < 60 ? String(format: "%.0f s left", eta) : String(format: "%.1f min left", eta / 60)) }
        return parts.joined(separator: " · ")
    }
}
