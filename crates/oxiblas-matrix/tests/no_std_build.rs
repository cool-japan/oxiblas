//! Regression guard for the advertised `no_std` support of `oxiblas-matrix`.
//!
//! A test binary always links `std` (the test harness and dev-dependencies
//! pull it into the crate graph), which makes `std`-only inherent float
//! methods (`f64::mul_add`, `f64::sqrt`, ...) and `std`-enabled dependency
//! features silently resolvable. That is how `oxiblas-core` (a dependency of
//! this crate) advertised `no_std` while failing to build on a real
//! bare-metal target. The build
//! check below therefore re-checks the library on its own, with
//! `--no-default-features`, on `thumbv7em-none-eabihf` (no `std` at all)
//! when that target is installed, so any `std` usage creeping back in fails
//! this test. The equivalent manual command is:
//!
//! ```text
//! cargo check -p oxiblas-matrix --lib --no-default-features --target thumbv7em-none-eabihf
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const EMBEDDED_TARGET: &str = "thumbv7em-none-eabihf";

/// Whether the rust-std component for `target` is installed.
fn target_installed(target: &str) -> bool {
    Command::new("rustc")
        .args(["--print", "target-libdir", "--target", target])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
        .is_some_and(|dir| dir.is_dir())
}

/// Runs `cargo check --lib --no-default-features` for this crate with the
/// given extra features, in a private target dir under Cargo's target tree.
fn check_without_std(features: &str, target_dir: &Path) -> TestResult {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(&manifest_dir)
        .args(["check", "--lib", "--no-default-features", "-j", "4"])
        .arg("--target-dir")
        .arg(target_dir);
    if !features.is_empty() {
        cmd.args(["--features", features]);
    }
    let embedded = target_installed(EMBEDDED_TARGET);
    if embedded {
        cmd.args(["--target", EMBEDDED_TARGET]);
    } else {
        eprintln!(
            "note: {EMBEDDED_TARGET} not installed; checking the no_std build on the host \
             only (install it with `rustup target add {EMBEDDED_TARGET}` for the real check)"
        );
    }
    // Keep the outer build's RUSTFLAGS (e.g. host `target-cpu`) out of the
    // cross check.
    cmd.env_remove("RUSTFLAGS");
    let output = cmd.output()?;
    assert!(
        output.status.success(),
        "`cargo check --lib --no-default-features --features '{features}'` (embedded target: \
         {embedded}) failed; std usage crept into the no_std build:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn test_no_std_library_builds_without_std() -> TestResult {
    let target_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("oxiblas_matrix_no_std_check");
    check_without_std("", &target_dir)
}
