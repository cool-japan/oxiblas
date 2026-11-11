//! Regression guard for the advertised `no_std` support of `oxiblas-core`.
//!
//! A test binary always links `std` (the test harness and dev-dependencies
//! pull it into the crate graph), which makes `std`-only inherent float
//! methods (`f64::mul_add`, `f64::sqrt`, ...) and `std`-enabled dependency
//! features silently resolvable. That is how `oxiblas-core` advertised
//! `no_std` while failing to build on a real bare-metal target. The build
//! check below therefore re-checks the library on its own, with
//! `--no-default-features`, on `thumbv7em-none-eabihf` (no `std` at all)
//! when that target is installed, so any `std` usage creeping back in fails
//! this test. The equivalent manual command is:
//!
//! ```text
//! cargo check -p oxiblas-core --lib --no-default-features --target thumbv7em-none-eabihf
//! cargo check -p oxiblas-core --lib --no-default-features --features f16,f128 \
//!     --target thumbv7em-none-eabihf
//! ```
//!
//! The second half of the file pins the other side of the contract: with the
//! default `std` feature the `Real`/`Field` float operations (which go through
//! `num_traits::Float`) must be bit-identical to the inherent `std` methods,
//! i.e. `num-traits/std` must keep taking precedence over its `libm` backend.

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
    let target_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("oxiblas_core_no_std_check");
    check_without_std("", &target_dir)?;
    check_without_std("f16,f128", &target_dir)
}

/// Deterministic xorshift64* generator.
#[cfg(feature = "std")]
struct XorShift(u64);

#[cfg(feature = "std")]
impl XorShift {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 / (1u64 << 53) as f64) * 20.0 - 10.0
    }
}

#[cfg(feature = "std")]
#[test]
fn test_std_float_ops_are_bit_identical_to_inherent_methods() {
    use oxiblas_core::scalar::{Field, Real};

    fn same64(what: &str, x: f64, got: f64, want: f64) {
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "f64 {what}({x}): {got:e} vs std {want:e}"
        );
    }
    fn same32(what: &str, x: f32, got: f32, want: f32) {
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "f32 {what}({x}): {got:e} vs std {want:e}"
        );
    }

    let mut rng = XorShift(0x0123_4567_89AB_CDEF);
    for _ in 0..2000 {
        let (x, y, z) = (rng.next_f64(), rng.next_f64(), rng.next_f64());
        same64("sqrt", x, Real::sqrt(x.abs()), x.abs().sqrt());
        same64("ln", x, Real::ln(x.abs()), x.abs().ln());
        same64("exp", x, Real::exp(x), x.exp());
        same64("sin", x, Real::sin(x), x.sin());
        same64("cos", x, Real::cos(x), x.cos());
        same64("atan2", x, Real::atan2(x, y), x.atan2(y));
        same64("powf", x, Real::powf(x.abs(), y), x.abs().powf(y));
        same64("mul_add", x, Real::mul_add(x, y, z), x.mul_add(y, z));
        same64("hypot", x, Real::hypot(x, y), x.hypot(y));
        same64("floor", x, Real::floor(x), x.floor());
        same64("round", x, Real::round(x), x.round());
        same64("powi", x, Field::powi(x, 7), x.powi(7));

        let (a, b, c) = (x as f32, y as f32, z as f32);
        same32("sqrt", a, Real::sqrt(a.abs()), a.abs().sqrt());
        same32("exp", a, Real::exp(a), a.exp());
        same32("sin", a, Real::sin(a), a.sin());
        same32("powf", a, Real::powf(a.abs(), b), a.abs().powf(b));
        same32("mul_add", a, Real::mul_add(a, b, c), a.mul_add(b, c));
        same32("hypot", a, Real::hypot(a, b), a.hypot(b));
        same32("powi", a, Field::powi(a, 5), a.powi(5));
    }
}
