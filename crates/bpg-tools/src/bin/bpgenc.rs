//! Public, reference-style BPG command-line entry point.

#[path = "../main.rs"]
mod app;

fn main() -> std::process::ExitCode {
    app::run()
}
