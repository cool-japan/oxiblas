//! CBLAS-compatible C ABI: basic Level 1, 2 and 3 routines.
//!
//! The `extern "C"` entry points in this module and its submodules follow the
//! semantics of reference CBLAS for valid calls. Invalid arguments are a silent
//! no-op: there is no `cblas_xerbla` (see [`crate::cblas`] for the other
//! deviations from reference CBLAS).
//!
//! # Layout
//!
//! CBLAS supports both row-major and column-major layouts. This library
//! uses column-major (Fortran) layout internally, so row-major operations
//! are converted using the identity: op(A) in row-major = op(A^T) in column-major.
//!
//! # Performance
//!
//! The wrappers validate their arguments (invalid input is a silent no-op) and
//! then either call a native kernel or run a reference-order scalar loop:
//!
//! - Level 1 (real): unit-stride calls go to the native slice kernels of
//!   [`crate::level1`] (SIMD where the kernel has one); any other increment,
//!   negative and zero included, is a scalar loop over the raw pointers.
//! - `gemv`: a scalar loop in the wrapper; the native `level2::gemv` is not
//!   used.
//! - Real `gemm`: the native `level3::gemm` when neither operand is
//!   transposed, a scalar loop otherwise.
//! - Complex dot products and `gemm`: scalar loops.
//!
//! Nothing here gathers or scatters vectors: strided data is read and written
//! in place.
//!
//! Split into per-area submodules (2026-08 hygiene pass) to keep every file
//! under the workspace's 2000-line limit; `vector_start_offset` below is a
//! shared helper used by several of them, so it stays here rather than
//! moving into any one of them.

/// Reference-BLAS start offset (in elements) for a strided vector of length `n`.
///
/// The BLAS contract lets an increment be negative, which means the vector is
/// traversed physically back-to-front: logical element `i` (`0..n`) lives at
/// byte offset `((1 - n) + i) * inc`. The walk therefore starts at the
/// physically-last element `(n - 1) * |inc|` and steps by `inc` (negative) each
/// iteration. For `inc >= 0` the start offset is simply `0`.
///
/// # Why isize
///
/// The whole computation stays in `isize` so that `(1 - n) * inc` (a large
/// non-negative value when `inc < 0`) never underflows the way the naive
/// `i * inc as usize` pattern does — that pattern computes an astronomically
/// large `usize` and reads far out of bounds, which is the UB this helper
/// exists to prevent.
#[inline]
fn vector_start_offset(n: usize, inc: isize) -> isize {
    if inc < 0 { (1 - n as isize) * inc } else { 0 }
}

mod complex_ops;
mod level1_ops;
mod level2_ops;
mod level3_ops;

pub use complex_ops::*;
pub use level1_ops::*;
pub use level2_ops::*;
pub use level3_ops::*;

#[cfg(test)]
mod tests;
