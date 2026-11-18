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
//!   feature, because the backend as it is here has run only under emulation
//!   (QEMU). On SME2 hardware only an early state has run: the unit tests of
//!   the GEMM macro-kernels, before the backend was connected to the public
//!   entry points, on an Apple M4 Pro (reported by tesch1 in PR #20). They
//!   passed but for the tests of the FPSR after a kernel call; "The frame" in
//!   `asm.rs` says what that run showed about the flags on macOS.
//! - When the backend is active ([`sme2_active`]), the public GEMM and GEMV
//!   entry points use it:
//!   - **GEMM.** The hook in `gemm_core` sits inside the blocked route, after
//!     the route policy chose that route and after the call's blocks were
//!     computed, and serves only the plans of the public entry points (no
//!     forced route, scope or micro-kernel). Every `f32` / `f64` call that the
//!     NEON build runs on the blocked route goes to the driver
//!     (`driver::try_gemm`) with the same blocks; the driver takes it (the
//!     blocked or the few-rows route of `driver.rs`, split over the SME units
//!     for a large parallel call) or declines it, and then NEON runs it. Not
//!     on SME2: the small routes (`m·n·k` up to 32³, and what the policy sends
//!     to them), thin products (`m == 1` or `n == 1`, unless the policy's thin
//!     route takes them, below), the driver's small class (off,
//!     `driver::SMALL_CLASS_ON`), a degenerate explicit KC, fully strided
//!     operands, complex element types.
//!   - **GEMV.** `gemv` and `gemv_with_par` (sequential and parallel paths)
//!     call the SME2 GEMV (`level2::gemv::sme2::try_gemv`) after `y` has been
//!     scaled by beta and after the `alpha == 0` return, before the NEON
//!     kernels: `f32` / `f64` with adjacent rows of A, past its size
//!     crossover. The thin GEMM route (where the policy turns it on: the
//!     default policy of an aarch64 build without `force-scalar` does for the
//!     thin products of its measured scope, both operands stored
//!     column-major on the NEON tier, see `gemm_policy`) gives its one GEMV
//!     to the same function. A thin product outside that scope (a transposed
//!     or strided view as an operand) takes the route of 0.2.3: the blocked
//!     route above 32³, whose driver declines the thin class, so it runs on
//!     NEON.
//!   - A call the backend does not take, or takes from a thread that is not
//!     ready, runs on NEON.
//! - **The promise.** Every result computed with the backend active equals
//!   the result of the same call in a build without it (a NEON build) bit for
//!   bit, except the sign and payload of NaN results (the ZA instructions
//!   return the default NaN, so even a NaN computed from a single NaN operand
//!   can change its sign and payload; NaN positions are equal) and the
//!   floating-point flags (the ZA instructions raise none). No result depends
//!   on whether the calling thread was ready, on the number of parts or on
//!   the policy's choices.
//! - Even when compiled in, the backend stays off at run time when the build
//!   caps SIMD ([`SME2_ALLOWED`]), when `OXIBLAS_DISABLE_SME2` is set to
//!   anything other than an empty string or `0`, when the operating system
//!   does not report SME2 with a 512-bit SVL, and when the first-use checks
//!   of the kernels fail on this CPU (see [`detect`]).
//! - **First use.** The first [`sme2_active`] call, or the first GEMM on the
//!   blocked route (more than 32³ multiply-adds) or GEMV past the crossover,
//!   on a thread that is ready for SME code runs the detection (on macOS one
//!   `/usr/sbin/sysctl` subprocess) and then the first-use checks of the
//!   macro-kernels, the ZA transposers, the few-rows GEMM and the GEMV
//!   against plain Rust, once per process: 2 084 kernel calls with `f64`
//!   (754 with `f32` alone) and about 91 MB of allocation traffic, 0.4 to
//!   0.65 s under QEMU (not measured on hardware yet). The backend counts as
//!   present only after all of them passed; a failure turns it off for the
//!   life of the process and writes one line to standard error. A GEMM or
//!   GEMV call never waits for another thread's detection (the subprocess
//!   or the checks): it runs on NEON then (see [`detect`], "Waiting, and
//!   fork"); `sme2_active` waits for the answer. A program that forks should
//!   call `sme2_active` once before it forks, from a thread that is ready
//!   for SME code (outside SME code, with a 512-bit streaming vector
//!   length): in a child forked while another thread was inside the
//!   detection or the checks, before they stored their result, GEMM and GEMV
//!   never use SME2 and never wait, and `sme2_active` would wait forever.
//! - `OXIBLAS_DISABLE_SME2` is read once per process, at the first use of the
//!   backend (the first [`sme2_active`] call or SME2 dispatch decision);
//!   changing it later has no effect.
//! - On macOS the operating-system report comes from one `/usr/sbin/sysctl`
//!   subprocess. Where it cannot be spawned (a sandbox that forbids process
//!   creation, a process that ignores `SIGCHLD` so the child cannot be
//!   waited for) the report is missing and SME2 stays off, so GEMM and GEMV
//!   keep running on NEON.
//! - The module holds the run-time detection with the public query
//!   [`sme2_active`], the streaming-region macro, the two macro-kernels, the
//!   operand packers (plain and through ZA), the few-rows GEMM, the scalar
//!   oracle with the first-use self-tests, and the GEMM driver with its split
//!   over SME units (see the table below); the SME2 GEMV lives in
//!   `level2/gemv/sme2.rs`.
//!
//! # Layout
//!
//! | File | Contents |
//! |---|---|
//! | `detect.rs` | run-time detection in two stages (the candidate: environment override, operating-system report, register probe; then the first-use checks of the kernels, read in a waiting form by `sme2_active` and the test gates and in a non-waiting form by the GEMM and GEMV paths), the SME unit count, the per-thread readiness check `thread_ready` and its rule |
//! | `fpstate.rs` | FPCR / FPSR access, and the default floating-point state the first-use checks run in |
//! | `asm.rs` | `sme_kernels!`, the only way to write a streaming region, and its fixed frame |
//! | `args.rs` | `MacroArgs`, the epilogue modes, the kernel contract (and, in test builds, its executable reference) |
//! | `kernel_f32.rs`, `kernel_f64.rs` | the macro-kernels: C tiles of 64 × 16 (`f32`) and 64 × 8 (`f64`) in ZA; the template of `kernel_f64.rs` is generated by `tools/gen_sme2_kernel_f64.py` |
//! | `pack.rs` | operand normalisation and the plain packers |
//! | `pack_za.rs` | the ZA transposers that pack `op(B) = B` and `op(A) = Aᵀ` blocks, and their first-use check |
//! | `few_rows.rs` | GEMM for few rows of C (`f32` up to 48, `f64` up to 32) in one streaming region, with its first-use check; the kernels are generated by `tools/gen_sme2_few_rows.py` |
//! | `driver.rs`, `parallel.rs` | the GEMM driver `try_gemm` that the hook in `gemm_core` calls (the decision, the policy, the sequential driver) and its split over SME units, each part checking its own thread and falling back to NEON |
//! | `oracle.rs`, `selftest.rs` | scalar emulation of the NEON arithmetic, the first-use self-test of the macro-kernels |
//! | `lint.rs`, `lint_scan.rs` (tests) | template lint, source scan against the registry |
//! | `test_support.rs` (tests) | hardware and backend gates with their marker lines, kernel and decision counters (quiet while the first-use check runs), force-off, FPSR and SME-state access |
//! | `test_sme_state.rs` (tests) | switches of a thread's SME state (ZA storage on, TPIDR2_EL0 set) for the readiness tests |
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
pub(crate) mod detect;
mod driver;
mod few_rows;
mod fpstate;
mod kernel_f32;
mod kernel_f64;
mod oracle;
mod pack;
mod pack_za;
mod parallel;
mod selftest;

pub(in crate::level3) use driver::try_gemm;

#[cfg(test)]
mod lint;
#[cfg(test)]
mod lint_scan;
#[cfg(test)]
mod test_interpreter;
#[cfg(test)]
mod test_interpreter_f64;
#[cfg(test)]
mod test_sme_state;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests_asm_sites;
#[cfg(test)]
mod tests_contract;
#[cfg(test)]
mod tests_few_rows;
#[cfg(test)]
mod tests_few_rows_grid;
#[cfg(test)]
mod tests_gemm;
#[cfg(test)]
mod tests_gemm_harness;
#[cfg(test)]
mod tests_gemm_policy;
#[cfg(test)]
mod tests_gemm_policy_parts;
#[cfg(test)]
mod tests_hook;
#[cfg(test)]
mod tests_hook_declines;
#[cfg(test)]
mod tests_hook_first_use;
#[cfg(test)]
mod tests_hook_grid;
#[cfg(test)]
mod tests_hook_thin;
#[cfg(test)]
mod tests_kernel_f32;
#[cfg(test)]
mod tests_kernel_f32_online;
#[cfg(test)]
mod tests_kernel_f64;
#[cfg(test)]
mod tests_kernel_f64_online;
#[cfg(test)]
mod tests_oracle;
#[cfg(test)]
mod tests_pack_za;
#[cfg(test)]
mod tests_pack_za_native;
#[cfg(test)]
mod tests_selftest;
#[cfg(test)]
mod tests_selftest_edits;
#[cfg(test)]
mod tests_selftest_online;
#[cfg(test)]
mod tests_selftest_paths;
#[cfg(test)]
mod tests_selftest_text;
#[cfg(test)]
mod tests_selftest_text_online;
#[cfg(test)]
mod tests_smoke;
#[cfg(test)]
mod tests_stress;
#[cfg(test)]
mod tests_threads;

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
/// - the first-use checks of the SME2 kernels passed on this CPU;
/// - the calling thread is not in streaming mode, holds no live ZA state,
///   has no pending lazy ZA save and still has a 512-bit streaming vector
///   length (Linux lets a thread change it), so a call from inside SME code
///   returns `false`. Such a thread is "ready" below.
///
/// A `true` result covers `f32`; `f64` additionally needs FEAT_SME_F64F64.
///
/// Detection runs once per process (on macOS it spawns `/usr/sbin/sysctl`
/// once). The first call from a thread that is ready for SME code then runs
/// the first-use checks: the SME2 kernels against plain Rust (2 084 kernel
/// calls with `f64`, 754 without, with FPCR = 0 and the caller's FPCR and
/// FPSR restored afterwards). The same happens at the first GEMM or GEMV
/// call that the backend could serve, whichever comes first. With the
/// backend active, GEMM and GEMV give the results of a build without it
/// bit for bit, except the sign and payload of NaN results and the
/// floating-point flags (see the module docs). When a check fails, the
/// backend stays off for the life of the process (this function returns
/// `false` from then on) and one line naming the failed check goes to
/// standard error: the kernels do not compute what they must on this CPU, so
/// the NEON kernels are used instead of returning wrong numbers. Later calls
/// only read three registers of the calling thread, and they execute no SME
/// instruction at all on CPUs without SME.
///
/// This function waits for the settled answer: while another thread runs
/// the detection (on macOS it waits for its `sysctl` subprocess), a call
/// waits for it, and while another thread runs the first-use checks, a call
/// waits for it too, from a ready thread and from one that is not ready
/// alike, and then answers what the checks settled. Only a ready thread
/// runs the checks itself; a call from a thread that is not ready while
/// nobody has run them returns `false` and leaves them to a later call from
/// a ready thread. So when this function returns, no detection or check that
/// was in progress when it was called is still in progress. Unless the
/// build caps SIMD (it then runs neither), the detection is settled; when
/// it found SME2 that the backend can use (a 512-bit streaming vector
/// length, not turned off by `OXIBLAS_DISABLE_SME2`: a candidate in the
/// words of `detect.rs`), the checks are settled too, unless they have not
/// run yet (only possible for a call from a thread that is not ready).
/// Where the detection found none, the checks never run and their lock is
/// never taken.
///
/// The GEMM and GEMV entry points never wait for another thread: a call that
/// finds another thread running the detection or the checks is computed on
/// NEON, which gives the same result. Both the detection and the checks
/// store their result before they release their lock, and a stored result
/// is read without it. A child forked while another thread holds one of
/// those locks, before that stage stored its result, inherits the lock in
/// its locked state: in that child the GEMM and GEMV calls never use SME2
/// and never wait, and this function would wait forever. A program that
/// forks should therefore call this function once before it forks:
///
/// - from a ready thread (outside SME code, with a 512-bit streaming vector
///   length; being outside SME code alone is not enough), it returns with
///   the detection and the checks settled as far as this process runs them
///   at all (see above), and the child inherits their results without ever
///   waiting for an inherited lock of the detection, whatever other threads
///   do;
/// - from a thread that is not ready, the checks may not have run yet. Their
///   lock is then free at the fork when no other thread is in a GEMM, GEMV
///   or detection call, or starts one, between this call and the fork (the
///   usual rule for forking a multi-threaded process), and the child's first
///   such call from a ready thread runs the checks itself.
pub fn sme2_active() -> bool {
    // The waiting form first: it settles the verdict (or waits for the
    // thread that settles it), so the non-waiting questions after it read a
    // settled verdict. A build that caps SIMD never runs the detection.
    SME2_ALLOWED && detect::hw().is_some() && detect::enabled::<f32>() && detect::thread_ready()
}
