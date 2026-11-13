//! Scalar emulation of the SME2 macro-kernel contract: the oracle.
//!
//! [`oracle_block`] computes, for one argument block of `args.rs`, the value
//! the kernel contract assigns to every element of `C[0..m, 0..n]` and writes
//! it there. It reads A, B and C through exactly the layouts of `args.rs`
//! (strides in elements, `MR = 64` row panels of A, `NR`-column panels of B,
//! column-major C), so it has the signature of a macro-kernel and can stand in
//! for one: the first-use self-test (`selftest.rs`) runs a real kernel and the
//! oracle on the same block and compares the results.
//!
//! It is library code, not test code, because the self-test runs in release
//! builds at the first use of the backend. Its tests (`tests_oracle.rs`) prove
//! it equal, bit for bit, to `args::reference`, the executable definition of
//! the contract in test builds, which `tests_contract.rs` in turn proves
//! equal to the NEON blocked path.
//!
//! # The arithmetic
//!
//! Per element, with `NR = T::NR`:
//!
//! ```text
//! acc = +0
//! for p in 0..kc (ascending): acc = fma(A(i, p), B(p, j), acc)     one rounding each
//! M0: C = acc                         C is not read
//! M1: C = acc + C
//! M2: C = acc * alpha                 C is not read
//! M3: t = (alpha == 1) ? acc : acc * alpha
//!     C = (beta == 1) ? t + C : fma(C, beta, t)
//! ```
//!
//! # What the oracle assumes about the floating-point environment
//!
//! The oracle's operations are ordinary Rust `f32`/`f64` arithmetic and
//! `mul_add`, so they run under the thread's FPCR as Rust assumes it: the
//! default environment, round to nearest (ties to even), no flush to zero
//! (`FPCR.FZ = 0`), the IEEE NaN rules (`FPCR.AH = 0`, `FPCR.DN = 0`).
//!
//! The kernels' arithmetic runs under the same FPCR with one difference. The
//! Arm architecture evaluates FMOPA and the ZA-array FADD/FMLA through the
//! shared pseudocode functions `FPMulAdd_ZA` and `FPAdd_ZA` ("SME ZA-targeting
//! floating-point behaviors"). Both take the caller's FPCR, set
//! `fpcr.DN = '1'` (default NaN) and `fpexc = FALSE` (no exception flags), and
//! pass everything else through: the rounding mode, `FZ` and `AH` are those of
//! the thread, so with the default FPCR the ZA instructions round exactly like
//! `mul_add` and `+`, subnormal inputs and outputs included. The SVE vector
//! arithmetic of the epilogues (FMUL, FMLA) uses the FPCR unchanged.
//! Consequences:
//!
//! - Finite and infinite results are compared bit for bit, signed zeros and
//!   subnormals included.
//! - NaN results are compared by position only: the ZA instructions return
//!   the default NaN where Rust propagates an input NaN's payload
//!   ([`same_bits_or_nan`]).
//! - The oracle raises its own floating-point flags (inexact and the like) on
//!   the calling thread; flags are not part of the contract.
//! - A thread that runs with a non-default FPCR (flush to zero, another
//!   rounding mode, `AH = 1`) is outside the contract: Rust itself does not
//!   support such a thread, and both the NEON path and the kernels would
//!   follow that FPCR while the oracle's results would not be comparable.
//!
//! # Mode values
//!
//! The mode is `args.mode` as [`Epilogue::select`](super::args::Epilogue::select)
//! encodes it (0 M0, 1 M1, 2 M2, 3 M3). A value above 3 is treated as M3, as
//! the executable definition in `args.rs` does; the contract never passes one.

// No `expect(dead_code)` here, unlike `selftest.rs` and `pack.rs`. Observed
// with rustc 1.89 and 1.95: the self-test's items are still reported dead
// (their `expect` is fulfilled), but what they call is not, so nothing in
// this file is reported dead before Wave 6 and an `expect` here would be
// unfulfilled (an error under `-D warnings`).

use super::args::{MR, MacroArgs, SmeElement};

/// Writes the contract's value into every element of `C[0..m, 0..n]` of
/// `args`, reading A, B and C through the layouts of `args.rs`, exactly as a
/// macro-kernel does (see the module docs for the arithmetic).
///
/// C is read only in modes M1 and M3; in M0 and M2 whatever C holds (a NaN
/// included) has no influence. Nothing outside `C[0..m, 0..n]` is read or
/// written, A rows `≥ m` and B columns `≥ n` are never read.
///
/// # Safety
///
/// As for a macro-kernel, minus the hardware: every address that the layout
/// of `args.rs` names for `args` must be valid for reads, namely A for
/// `i < m`, `p < kc` and B for `p < kc`, `j < n`; C for `i < m`, `j < n` must be
/// valid for reads and writes and must not overlap A or B. `m`, `n` or `kc`
/// may be zero (then the block is empty or every accumulator is `+0`).
pub(super) unsafe fn oracle_block<T: SmeElement>(args: &MacroArgs<T>) {
    let nr = T::NR;
    let alpha = T::from_bits64(args.alpha_bits);
    let beta = T::from_bits64(args.beta_bits);
    let alpha_one = alpha == T::ONE;
    let beta_one = beta == T::ONE;
    for j in 0..args.n {
        let b_column = (j / nr) * args.b_panel + j % nr;
        for i in 0..args.m {
            let a_row = (i / MR) * args.a_panel + i % MR;
            let mut acc = T::ZERO;
            for p in 0..args.kc {
                // SAFETY: `i < m` and `p < kc`, so this is the address of
                // `A(i, p)`, valid for reads by the caller's contract.
                let a = unsafe { *args.a.add(a_row + p * args.a_depth) };
                // SAFETY: `p < kc` and `j < n`, so this is the address of
                // `B(p, j)`, valid for reads by the caller's contract.
                let b = unsafe { *args.b.add(b_column + p * nr) };
                acc = a.fused(b, acc);
            }
            // SAFETY: `i < m` and `j < n`: the address of `C(i, j)`, inside
            // the block the caller guarantees.
            let c = unsafe { args.c.add(i + j * args.ldc) };
            let value = match args.mode {
                0 => acc,
                // SAFETY: `c` is `C(i, j)`, valid for reads; M1 reads C.
                1 => acc + unsafe { *c },
                2 => acc * alpha,
                _ => {
                    let t = if alpha_one { acc } else { acc * alpha };
                    // SAFETY: `c` is `C(i, j)`, valid for reads; M3 reads C.
                    let old = unsafe { *c };
                    if beta_one {
                        t + old
                    } else {
                        old.fused(beta, t)
                    }
                }
            };
            // SAFETY: `c` is `C(i, j)`, valid for writes and not aliased by A
            // or B (caller's contract).
            unsafe { *c = value };
        }
    }
}

/// The comparison of the promise for whole slices: `true` when both slices
/// have the same length and every pair of elements has equal bits or is a
/// pair of NaNs (NaN payloads may differ, NaN positions may not).
///
/// `-0` and `+0` differ; two empty slices are equal.
pub(super) fn same_bits_or_nan<T: SmeElement>(got: &[T], want: &[T]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.bits() == w.bits() || (g.nan() && w.nan()))
}
