//! `piv`: planar image velocimetry from high-speed video.

mod analyze;
mod info;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use piv_core::FrameRate;

#[derive(Parser)]
#[command(
    name = "piv",
    version,
    about = "Planar image velocimetry for high-speed light-sheet video"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print resolution, frame count, container and capture frame rates.
    Info {
        input: PathBuf,
        /// Override the capture frame rate (e.g. 240, 1000, 239.76, 240000/1001).
        #[arg(long)]
        capture_fps: Option<FrameRate>,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Compute displacement and velocity fields for frame pairs.
    Analyze(Box<analyze::AnalyzeArgs>),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Info {
            input,
            capture_fps,
            json,
        } => info::run(&input, capture_fps, json),
        Cmd::Analyze(args) => analyze::run(*args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
