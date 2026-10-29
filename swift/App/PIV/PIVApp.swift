import AppKit
import PIVKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct PIVApp: App {
    @State private var model: AppModel

    init() {
        let service: AnalysisService
        do {
            service = try PIVSession()
        } catch {
            service = UnavailableService(reason: error.localizedDescription)
        }
        _model = State(initialValue: AppModel(service: service))
    }

    var body: some Scene {
        Window("PIV", id: "main") {
            ContentView()
                .environment(model)
                .frame(minWidth: 1000, minHeight: 680)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open…") { openPanel(model) }
                    .keyboardShortcut("o")
                    .disabled(model.isRunning)
            }
        }
    }
}

/// Shows the system open panel and opens the chosen clip.
@MainActor
func openPanel(_ model: AppModel) {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.movie]
    panel.allowsMultipleSelection = false
    panel.message = "Choose a high-speed clip to analyse"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await model.open(url) }
}

/// Stands in when Metal isn't available, so the app can say why instead of crashing.
struct UnavailableService: AnalysisService {
    let reason: String
    func probe(_ url: URL) async throws -> MediaInfo { throw PIVError.metal(reason) }
    func analyze(_ request: AnalysisRequest, render: RenderOptions?) -> AsyncThrowingStream<AnalysisEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: PIVError.metal(reason)) }
    }
}

struct ContentView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Group {
            switch model.phase {
            case .empty: EmptyStateView()
            case .setup: SetupView()
            case .running(let status): RunningView(status: status)
            case .result(let output): ResultView(output: output)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first, !model.isRunning else { return false }
            Task { await model.open(url) }
            return true
        }
        .task {
            #if DEBUG
            SmokeTest.runIfRequested(model)
            #endif
        }
        .alert("PIV", isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.error ?? "")
        }
    }
}

struct EmptyStateView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "wind")
                .font(.system(size: 56, weight: .light))
                .foregroundStyle(.secondary)
            Text("Planar image velocimetry")
                .font(.title2)
            Text("Drop a high-speed light-sheet clip here, or open one.")
                .foregroundStyle(.secondary)
            Button("Open Clip…") { openPanel(model) }
                .keyboardShortcut(.defaultAction)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
