//! Opt-in SME2 backend for Arm CPUs with SME2 (Apple M4 and later).
//!
//! # What SME2 is used for
//!
//! The Scalable Matrix Extension (SME) adds a streaming execution mode and the
//! ZA array, a square register file of SVL × SVL bits that holds whole tiles
//! of a result matrix. One outer-product instruction (FMOPA) updates a full
//! ZA tile from two vectors, so a matrix product accumulates in ZA without
//! the per-element shuffling that NEON needs. SME2 adds multi-vector loads,
//! stores and arithmetic on groups of ZA vectors. On an Apple M4 each
//! performance-core cluster has one SME unit with a 512-bit streaming vector
//! length (SVL), which is the configuration this backend is written for.
//!
//! The backend serves real `f32` and `f64` level-3 products. Every streaming
//! region lives inside a single `asm!` block, because stable Rust has no
//! SME intrinsics and no `#[target_feature(enable = "sme")]`.
//!
//! # Status
//!
//! - The module is compiled only with the `sme2` Cargo feature on `aarch64`.
//!   The feature is off by default and is not part of the facade's `full`
//!   feature, because the backend has not been validated on SME2 hardware yet.
//! - Even when compiled in, the backend stays off at run time when the build
//!   caps SIMD ([`SME2_ALLOWED`]), when `OXIBLAS_DISABLE_SME2` is set to
//!   anything other than an empty string or `0`, and when the operating
//!   system does not report SME2 with a 512-bit SVL (see [`detect`]).
//! - `OXIBLAS_DISABLE_SME2` is read once per process, at the first use of the
//!   backend (the first [`sme2_active`] call or SME2 dispatch decision);
//!   changing it later has no effect.
//! - On macOS the operating-system report comes from one `/usr/sbin/sysctl`
//!   subprocess. Where it cannot be spawned (a sandbox that forbids process
//!   creation, a process that ignores `SIGCHLD` so the child cannot be
//!   waited for) the report is missing and SME2 stays off, so GEMM keeps
//!   running on NEON.
//! - The module currently provides run-time detection and the public query
//!   [`sme2_active`], plus the foundation of the kernels: the streaming-region
//!   macro (`asm.rs`), the argument block and kernel contract (`args.rs`), the
//!   static template lint and source scan (tests). GEMM still runs on the
//!   NEON kernels.
//!
//! # Layout
//!
//! | File | Contents |
//! |---|---|
//! | `detect.rs` | run-time detection, register probe, environment override |
//! | `asm.rs` | `sme_kernels!`, the only way to write a streaming region |
//! | `args.rs` | `MacroArgs`, the epilogue modes, the kernel contract |
//! | `kernel_f32.rs`, `kernel_f64.rs` | the 64 × 16 / 64 × 8 macro-kernels |
//! | `pack.rs` | operand normalisation and plain packers |
//! | `driver.rs`, `parallel.rs` | the GEMM driver and its split over SME units |
//! | `oracle.rs`, `selftest.rs` | scalar emulation of the NEON arithmetic, first-use self-test |
//! | `lint.rs`, `lint_scan.rs` (tests) | template lint, source scan against the registry |
//! | `test_support.rs` (tests) | hardware gate, counters, force-off, FPSR access |
//! | `tests_*.rs` | tests |
//!
//! # Acknowledgment
//!
//! The SME2 backend follows the design that tesch1 proposed in PR #20 and
//! issues #21 and #22 (the MTGEMM-A approach: C tiles kept in ZA with the
//! packed B panel as the outer-product row operand, predicated macro-kernels
//! that run a whole block in one streaming region, transposition through ZA,
//! one part per SME unit), together with their Apple M4 Pro measurements.
//! OxiBLAS's implementation was written independently from that description
//! and the Arm architecture documentation and contains no code from PR #20 or
//! MTGEMM-A.

pub(crate) mod args;
pub(crate) mod asm;
mod detect;
mod driver;
mod kernel_f32;
mod kernel_f64;
mod oracle;
mod pack;
mod parallel;
mod selftest;

#[cfg(test)]
mod lint;
#[cfg(test)]
mod lint_scan;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests_contract;
#[cfg(test)]
mod tests_gemm;
#[cfg(test)]
mod tests_kernel_f32;
#[cfg(test)]
mod tests_kernel_f64;
#[cfg(test)]
mod tests_oracle;
#[cfg(test)]
mod tests_selftest;
#[cfg(test)]
mod tests_smoke;
#[cfg(test)]
mod tests_stress;

/// Whether the build lets the SME2 backend run at all.
///
/// `false` when the crate is built with `force-scalar`, `max-simd-128` or
/// `max-simd-256`: SME2's 512-bit streaming vectors exceed every one of those
/// caps, so the backend stays off at run time. It is a constant rather than a
/// `cfg` so that such builds (`--all-features` turns on `force-scalar`) still
/// compile and lint the whole module.
const SME2_ALLOWED: bool = !cfg!(any(
    feature = "force-scalar",
    feature = "max-simd-128",
    feature = "max-simd-256"
));

/// Returns `true` when the SME2 backend is active for calls made from the
/// current thread.
///
/// This is the run-time half of the opt-in `sme2` Cargo feature. It is `true`
/// only when all of the following hold:
///
/// - the crate was built without `force-scalar`, `max-simd-128` and
///   `max-simd-256`, which cap SIMD below SME2's 512-bit vectors;
/// - the `OXIBLAS_DISABLE_SME2` environment variable is unset, empty or `0`;
/// - the operating system reports FEAT_SME and FEAT_SME2 (macOS through
///   `sysctl hw.optional.arm.*`, Linux and Android through `AT_HWCAP2`), and
///   the streaming vector length is 512 bits;
/// - the calling thread is not in streaming mode, holds no live ZA state and
///   has no pending lazy ZA save, so a call from inside SME code returns
///   `false`.
///
/// A `true` result covers `f32`; `f64` additionally needs FEAT_SME_F64F64.
///
/// Detection runs once per process (on macOS it spawns `/usr/sbin/sysctl`
/// once). Later calls only read three registers of the calling thread, and
/// they execute no SME instruction at all on CPUs without SME.
pub fn sme2_active() -> bool {
    detect::enabled::<f32>() && detect::thread_ready()
}
