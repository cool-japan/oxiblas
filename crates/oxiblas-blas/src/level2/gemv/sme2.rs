//! SME2 GEMV kernels for column-major `f32` / `f64` matrices on Arm CPUs
//! with SME2 (the opt-in `sme2` feature; see `level3::sme2`).
//!
//! [`try_gemv`] computes `y += alpha·op(A)·x` on SME2 or declines without
//! touching `y`. Both transposes are defined to give the results of the NEON
//! kernels of `neon.rs` bit for bit (NaN sign and payload aside: the ZA
//! instructions return the default NaN), so routing a call through SME2
//! never changes a result. The GEMV drivers of `mod.rs` (`gemv`, and
//! `gemv_with_par` on its sequential and its parallel path) call it after
//! `y` has been scaled by beta and after their `alpha == 0` return, before
//! their NEON kernels; the thin GEMM route (`level3/gemm_thin.rs`) calls it
//! for its one GEMV on the column-major operand. The detection runs
//! [`self_check`] as one of its first-use checks.
//!
//! # The decision (on the calling thread; nothing in it writes `y`)
//!
//! 1. The crossover ([`crossover`], provisional constants for M4 tuning):
//!    `m >= `[`MIN_ROWS`]` && m·n >= `[`MIN_WORK`] for the stored `m × n`
//!    matrix, and for `Trans` / `ConjTrans` also `n >= `[`TRANS_MIN_COLS`].
//!    Below it the 28 ns streaming entry and exit dominate. It comes first,
//!    as plain compares of the dimensions in the inlined entry: a GEMV below
//!    it pays only those compares and never starts the detection (on macOS a
//!    subprocess, then the first-use checks).
//! 2. `T` is `f32` or `f64` (the `TypeId` hand-off of `level1/typed.rs`;
//!    `try_gemv` has the bounds of the generic `gemv`, no `Send` or `Sync`,
//!    and the split over threads works on the concrete type) and
//!    `detect::enabled::<T>()` (so a build that caps SIMD below SME2, the
//!    environment switch and a test's force-off all decline here). It never
//!    waits for another thread: while another thread runs the detection
//!    (the subprocess on macOS) or the first-use checks it is `false` and
//!    the drivers' NEON kernels compute the call (also for good in a child
//!    forked before they stored their result; `level3::sme2::detect`,
//!    "Waiting, and fork"). The first such call of the process may run the
//!    detection and the checks itself, on the calling thread. A decision
//!    that gets past it has seen the verdict settled, so no part runs the
//!    checks.
//! 3. `a.col_stride() == 1`, `x` and `y` have the lengths of `op(A)`, and
//!    `alpha != 0` (the drivers never send `alpha == 0`; declining it keeps
//!    the "`false` = untouched" promise trivially true). These are the
//!    operands the NEON dispatch of `mod.rs` takes (`f32` / `f64` with
//!    adjacent rows); every other one goes to the drivers' other code.
//! 4. `detect::thread_ready()` on the calling thread.
//! 5. The parts ([`part_count`]): 1, unless `par` is parallel, at least two
//!    SME units are reported (`detect::units`) and the matrix holds at least
//!    [`PAR_MIN_BYTES`]; then `min(units, par.num_threads())`
//!    (`num_threads` asked last: for `Par::Rayon` it builds rayon's global
//!    pool). `NoTrans` splits the rows in whole vector steps
//!    ([`SmeGemv::LANES`] rows), `Trans` the columns in groups of
//!    [`TRANS_SPLIT_COLS`], into near-equal non-empty runs ([`split`]):
//!    disjoint pieces of `y`.
//!
//! Then `test_support::note_decision()` counts the decision (test builds)
//! and the call returns `true`. Every part checks `detect::thread_ready()`
//! on the thread that runs it, and runs the NEON kernel of `neon.rs` on its
//! sub-view when that thread is not ready: the same bits. A part never asks
//! `detect::enabled` again (the test force-off switch is thread-local).
//!
//! # NoTrans: `y[i] = fma(A[i, j], t_j, y[i])` for `j` ascending
//!
//! With `t_j = alpha · x[j]` (one rounded multiplication), exactly what the
//! NEON kernel does for every row. With a 512-bit streaming vector length
//! the ZA array is 64 vectors of 64 bytes; a `vgx4` group `g` (0..16) is
//! the four ZA vectors `g, g + 16, g + 32, g + 48`. The kernel holds a block
//! of up to 1024 `f32` / 512 `f64` rows of `y` in the 16 groups (group `g`
//! takes rows `r0 + 64g .. r0 + 64g + 64` for `f32`, 32 rows for `f64`, in
//! that register order), and sweeps the columns in ascending order, four
//! (`f32`) or two (`f64`) at a time: one `ld1rqw` / `ld1rqd` of the `x`
//! values replicated to every 128-bit segment, one unpredicated SVE `fmul`
//! by alpha, then for every group one four-vector load of each column and
//! one indexed multi-vector `fmla za.s[w8, 0, vgx4], {…}, z8.s[c]` per
//! column, columns in order. The last 1–3 (`f32`) or 1 (`f64`) columns
//! take the same path with a predicated `ld1rq*` and one FMLA per live
//! column only (never a padded FMLA: `fma(A, +0, -0)` would turn a `-0` of
//! `y` into `+0`). Every load and store of `A` and `y` is predicated by a
//! `whilelt …, vlx4` counter against `m` (`/z` loads), so no row `>= m` is
//! read or written; the `ld1rq*` of `x` is predicated against `n`.
//!
//! # Trans: lane partials in ZA, finished in Rust
//!
//! For column `j`, with `V` = 16 (`f32`) / 8 (`f64`) lanes and
//! `body = m - m % V`, the NEON kernel accumulates `V` chains (lane `l`
//! takes rows `i0 + l` of every full `V`-row step), then a vector tail of
//! whole 8 / 4 (`f32`) or 4 / 2 (`f64`) rows, reduces the lanes pairwise,
//! adds the scalar tail with `mul_add` and finally `y[j] += alpha · s`. The
//! kernel keeps one ZA vector per column (column `4g + k` of a 64-column
//! block in ZA vector `g + 16k`, so the four columns of group `g` are the
//! vector group `g`), feeds it with strided four-vector loads (four row
//! steps of one column) and multi-and-single FMLAs with the matching `x`
//! vector, then the remaining whole `V`-row steps with single-vector loads,
//! then the vector tail as ONE step with a lane predicate ([`vector_tail`]):
//! `f32` rows `body..` go to lanes `0..8` when `r = m % 16 >= 8` and to
//! lanes `8..12` (loaded from base `body - 8`) for the next four rows when
//! there are four more; `f64` likewise with 4 / 2 rows. A and `x` are loaded
//! with the same predicate, `/z`, and the dead lanes of `x` are then
//! negated, so a dead lane adds `+0 · -0 = -0`, which changes no
//! accumulator. (`+0 · +0` would not do: a lane is `-0` when its products
//! underflow to `-0`, `-0 + +0` is `+0`, and NEON never touches that lane,
//! so the sign of a zero dot product, and of `y`, would differ.) A column
//! past `n` of the last group is loaded with an empty
//! predicate (nothing read) and its partials are never stored. The kernel
//! stores the `V` lane partials of every column to a scratch buffer (one
//! streaming region per at most [`TRANS_COLS_PER_CALL`] columns, a
//! thread-local buffer reused across calls), and [`trans_finish`] finishes
//! exactly as NEON does.
//!
//! # Floating-point flags
//!
//! The ZA-array FMLA raises no flags and forces the default NaN (the Arm
//! ARM's `FPMulAdd_ZA`), so a kernel call raises only what the SVE `fmul`
//! of `alpha · x` raises (`NoTrans`), and the Rust finish what its scalar
//! arithmetic raises (`Trans`). Flag parity with NEON is not promised.
//!
//! The tests live in `sme2_tests.rs` and `sme2_tests_grid.rs`.
//!
//! # Acknowledgment
//!
//! The SME2 GEMV follows the design that tesch1 proposed in issue #21 (a
//! block of `y` held in ZA and updated with indexed multi-vector FMLA for
//! `NoTrans`, ZA accumulators fed by strided multi-vector loads for
//! `Trans`). OxiBLAS's implementation was written independently from that
//! description and the Arm architecture documentation and contains no code
//! from PR #20 or MTGEMM-A.

use core::mem::offset_of;
#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
#[cfg(test)]
use std::marker::PhantomData;
#[cfg(test)]
use std::sync::{Condvar, Mutex};
use std::thread::LocalKey;
#[cfg(test)]
use std::thread::ThreadId;
#[cfg(test)]
use std::time::{Duration, Instant};

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{MatMut, MatRef};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::GemvTrans;
use super::neon;
use crate::level1::typed::{same_matmut, same_matref, same_value};
use crate::level3::sme2::args::SmeElement;
use crate::level3::sme2::asm::sme_kernels;
use crate::level3::sme2::detect;

/// Smallest stored row count `m` that SME2 takes (crossover, provisional).
pub(super) const MIN_ROWS: usize = 64;

/// Smallest `m · n` that SME2 takes: `128²` (crossover, provisional).
pub(super) const MIN_WORK: usize = 128 * 128;

/// Smallest column count `n` that SME2 takes for `Trans` / `ConjTrans`
/// (crossover, provisional): fewer columns leave most of a 64-column ZA
/// block idle.
pub(super) const TRANS_MIN_COLS: usize = 16;

/// Matrix size from which a call is split over SME units: 8 MiB.
pub(super) const PAR_MIN_BYTES: usize = 8 << 20;

/// Columns of one `Trans` part boundary step: parts start at multiples of 4
/// columns (the column group of the kernel).
pub(super) const TRANS_SPLIT_COLS: usize = 4;

/// Columns per `Trans` kernel call: one streaming region per at most 1024
/// columns, whose lane partials fill the scratch buffer (64 KiB).
pub(super) const TRANS_COLS_PER_CALL: usize = 1024;

/// The argument block of the GEMV kernels. `#[repr(C)]`, every field 8
/// bytes, identical offsets for `f32` and `f64`. Sizes and strides are in
/// ELEMENTS.
///
/// | Field | NoTrans | Trans |
/// |---|---|---|
/// | `a` | `A(0, 0)`; `A(i, j) = a[i + j·lda]` | the same |
/// | `lda` | column distance, `>= 1` | the same |
/// | `x` | `x[0..n]` | `x[0..m]` |
/// | `out` | `y[0..m]` (read and written) | lane partials, `n · V` elements written |
/// | `m`, `n` | rows, columns, both `>= 1` | the same, `m >= 16` |
/// | `alpha_bits` | alpha's bit pattern (`f32` zero-extended) | unused |
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct GemvArgs<T> {
    /// `A(0, 0)`.
    pub(super) a: *const T,
    /// Elements between columns `j` and `j + 1` of A.
    pub(super) lda: usize,
    /// The vector `x`.
    pub(super) x: *const T,
    /// `y` (NoTrans) or the lane partials (Trans).
    pub(super) out: *mut T,
    /// Rows of the stored A.
    pub(super) m: usize,
    /// Columns of the stored A.
    pub(super) n: usize,
    /// alpha's bit pattern (NoTrans only).
    pub(super) alpha_bits: u64,
}

/// Offset of [`GemvArgs::a`].
pub(super) const OFF_GV_A: usize = offset_of!(GemvArgs<f32>, a);
/// Offset of [`GemvArgs::lda`].
pub(super) const OFF_GV_LDA: usize = offset_of!(GemvArgs<f32>, lda);
/// Offset of [`GemvArgs::x`].
pub(super) const OFF_GV_X: usize = offset_of!(GemvArgs<f32>, x);
/// Offset of [`GemvArgs::out`].
pub(super) const OFF_GV_OUT: usize = offset_of!(GemvArgs<f32>, out);
/// Offset of [`GemvArgs::m`].
pub(super) const OFF_GV_M: usize = offset_of!(GemvArgs<f32>, m);
/// Offset of [`GemvArgs::n`].
pub(super) const OFF_GV_N: usize = offset_of!(GemvArgs<f32>, n);
/// Offset of [`GemvArgs::alpha_bits`].
pub(super) const OFF_GV_ALPHA: usize = offset_of!(GemvArgs<f32>, alpha_bits);

// The kernels read both element types through the same offsets.
const _: () = {
    assert!(OFF_GV_A == offset_of!(GemvArgs<f64>, a));
    assert!(OFF_GV_LDA == offset_of!(GemvArgs<f64>, lda));
    assert!(OFF_GV_X == offset_of!(GemvArgs<f64>, x));
    assert!(OFF_GV_OUT == offset_of!(GemvArgs<f64>, out));
    assert!(OFF_GV_M == offset_of!(GemvArgs<f64>, m));
    assert!(OFF_GV_N == offset_of!(GemvArgs<f64>, n));
    assert!(OFF_GV_ALPHA == offset_of!(GemvArgs<f64>, alpha_bits));
    assert!(core::mem::size_of::<GemvArgs<f32>>() == 56);
    assert!(core::mem::size_of::<GemvArgs<f64>>() == 56);
};

sme_kernels! {
    /// `y[0..m] += alpha · A · x` for an `f32` column-major A in one
    /// streaming region (module docs: "NoTrans").
    ///
    /// Loop nest (labels in brackets; `G` = groups of the row block):
    ///
    /// ```text
    /// for r0 in (0..m).step_by(1024)                     [20] row block
    ///     for g in 0..16 while r0 + 64g < m             [21] y -> ZA group g
    ///     G = groups loaded                              [22]
    ///     for j0 in (0..n - n % 4).step_by(4)           [30] four columns
    ///         z8 = alpha · x[j0..j0+4] in every segment
    ///         for g in 0..G                              [31]
    ///             for c in 0..4: za[g] += A(rows of g, j0 + c) · z8[c]
    ///     if n % 4 != 0                                  [40] last columns
    ///         z8 = alpha · x[j0..n] (predicated), q = n - j0
    ///         for g in 0..G                              [41]
    ///             for c in 0..q: za[g] += A(rows of g, j0 + c) · z8[c]
    ///                                                    [42] next group
    ///     for g in 0..G: ZA group g -> y                 [50] [51]
    /// ```
    ///
    /// Register map:
    ///
    /// | Register | Use |
    /// |---|---|
    /// | `x1` | `a`, `A(0, 0)` |
    /// | `x2` | `lda` in bytes |
    /// | `x3` | `x` |
    /// | `x4` | `y` |
    /// | `x5`, `x6` | `m`, `n` |
    /// | `w7` | alpha bits |
    /// | `x8` / `w8` | group `g`, the ZA vector-group select |
    /// | `x9` | `r0`, first row of the row block |
    /// | `x10` | `r0 + 64g`, first row of group `g` |
    /// | `x12` | `G`, groups of the row block |
    /// | `x13` | `j0`, first column of the column group |
    /// | `x14`, `x20`, `x21`, `x22` | `A(0, j0 + c)` for `c` = 0..3 |
    /// | `x15` | live columns of the last group (1..3) |
    /// | `x16` | `j0 + 4` |
    /// | `z0`–`z3`, `z4`–`z7`, `z12`–`z15`, `z16`–`z19` | rows of group `g` of columns `j0 + c` (and `y` through ZA) |
    /// | `z8` | `alpha · x[j0..j0 + 4]`, replicated per 128-bit segment |
    /// | `z28` | alpha broadcast |
    /// | `p1` | columns `< n` of the column group (for `ld1rqw`) |
    /// | `pn9` | rows `< m` of group `g` (64-lane counter) |
    ///
    /// It reads `A(i, j)` and `y[i]` only for `i < m`, `j < n`, and `x[j]`
    /// only for `j < n`; it writes only `y[0..m]`. It never writes the
    /// argument block and leaves through the frame's exit label.
    ///
    /// # Safety
    ///
    /// - `detect::thread_ready()` must have returned `true` on this thread
    ///   (SME2 with a 512-bit streaming vector length, not in streaming
    ///   mode, no live ZA state, no pending lazy save).
    /// - `m >= 1`, `n >= 1`; `A(i, j)` at `a + i + j·lda` must be readable
    ///   for `i < m`, `j < n`, `x[0..n]` readable, `y[0..m]` readable and
    ///   writable, and `y` must not overlap A, `x` or the argument block.
    pub(super) streaming fn sme2_gemv_notrans_f32(args: &GemvArgs<f32>) {
        extensions: [],
        consts: [
            off_a = OFF_GV_A,
            off_lda = OFF_GV_LDA,
            off_x = OFF_GV_X,
            off_out = OFF_GV_OUT,
            off_m = OFF_GV_M,
            off_n = OFF_GV_N,
            off_alpha = OFF_GV_ALPHA,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_lda}]",
            "ldr x3, [x0, #{off_x}]",
            "ldr x4, [x0, #{off_out}]",
            "ldr x5, [x0, #{off_m}]",
            "ldr x6, [x0, #{off_n}]",
            "ldr w7, [x0, #{off_alpha}]",
            "lsl x2, x2, #2",
        ],
        body: [
            "dup z28.s, w7",
            "mov x9, #0",
            "20:",
            "mov x8, #0",
            "mov x10, x9",
            "21:",
            "whilelt pn9.s, x10, x5, vlx4",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x4, x10, lsl #2]",
            "mova za.d[w8, 0, vgx4], {{z0.d-z3.d}}",
            "add x8, x8, #1",
            "add x10, x10, #64",
            "cmp x8, #16",
            "b.hs 22f",
            "cmp x10, x5",
            "b.lo 21b",
            "22:",
            "mov x12, x8",
            "mov x13, #0",
            "mov x14, x1",
            "30:",
            "add x16, x13, #4",
            "cmp x16, x6",
            "b.hi 40f",
            "whilelt p1.s, x13, x6",
            "ld1rqw {{z8.s}}, p1/z, [x3, x13, lsl #2]",
            "fmul z8.s, z8.s, z28.s",
            "add x20, x14, x2",
            "add x21, x20, x2",
            "add x22, x21, x2",
            "mov x8, #0",
            "mov x10, x9",
            "31:",
            "whilelt pn9.s, x10, x5, vlx4",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x14, x10, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z0.s-z3.s}}, z8.s[0]",
            "ld1w {{z4.s-z7.s}}, pn9/z, [x20, x10, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z4.s-z7.s}}, z8.s[1]",
            "ld1w {{z12.s-z15.s}}, pn9/z, [x21, x10, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z12.s-z15.s}}, z8.s[2]",
            "ld1w {{z16.s-z19.s}}, pn9/z, [x22, x10, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z16.s-z19.s}}, z8.s[3]",
            "add x8, x8, #1",
            "add x10, x10, #64",
            "cmp x8, x12",
            "b.lo 31b",
            "mov x13, x16",
            "add x14, x14, x2, lsl #2",
            "b 30b",
            "40:",
            "cmp x13, x6",
            "b.hs 50f",
            "whilelt p1.s, x13, x6",
            "ld1rqw {{z8.s}}, p1/z, [x3, x13, lsl #2]",
            "fmul z8.s, z8.s, z28.s",
            "sub x15, x6, x13",
            "add x20, x14, x2",
            "add x21, x20, x2",
            "mov x8, #0",
            "mov x10, x9",
            "41:",
            "whilelt pn9.s, x10, x5, vlx4",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x14, x10, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z0.s-z3.s}}, z8.s[0]",
            "cmp x15, #2",
            "b.lo 42f",
            "ld1w {{z4.s-z7.s}}, pn9/z, [x20, x10, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z4.s-z7.s}}, z8.s[1]",
            "cmp x15, #3",
            "b.lo 42f",
            "ld1w {{z12.s-z15.s}}, pn9/z, [x21, x10, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z12.s-z15.s}}, z8.s[2]",
            "42:",
            "add x8, x8, #1",
            "add x10, x10, #64",
            "cmp x8, x12",
            "b.lo 41b",
            "50:",
            "mov x8, #0",
            "mov x10, x9",
            "51:",
            "whilelt pn9.s, x10, x5, vlx4",
            "mova {{z0.d-z3.d}}, za.d[w8, 0, vgx4]",
            "st1w {{z0.s-z3.s}}, pn9, [x4, x10, lsl #2]",
            "add x8, x8, #1",
            "add x10, x10, #64",
            "cmp x8, x12",
            "b.lo 51b",
            "add x9, x9, #1024",
            "cmp x9, x5",
            "b.lo 20b",
        ],
    }

    /// `y[0..m] += alpha · A · x` for an `f64` column-major A in one
    /// streaming region (module docs: "NoTrans"): the `f32` kernel with
    /// 8-lane vectors, so a group holds 32 rows, a row block 512 rows, and
    /// the columns go two at a time (`ld1rqd`, lane indices 0 and 1).
    ///
    /// Loop nest (labels in brackets; `G` = groups of the row block):
    ///
    /// ```text
    /// for r0 in (0..m).step_by(512)                      [20] row block
    ///     for g in 0..16 while r0 + 32g < m             [21] y -> ZA group g
    ///     G = groups loaded                              [22]
    ///     for j0 in (0..n - n % 2).step_by(2)           [30] two columns
    ///         z8 = alpha · x[j0..j0+2] in every segment
    ///         for g in 0..G                              [31]
    ///             for c in 0..2: za[g] += A(rows of g, j0 + c) · z8[c]
    ///     if n is odd                                    [40] last column
    ///         z8 = alpha · x[j0] (predicated)
    ///         for g in 0..G: za[g] += A(rows of g, j0) · z8[0]   [41]
    ///     for g in 0..G: ZA group g -> y                 [50] [51]
    /// ```
    ///
    /// Register map:
    ///
    /// | Register | Use |
    /// |---|---|
    /// | `x1` | `a`, `A(0, 0)` |
    /// | `x2` | `lda` in bytes |
    /// | `x3` | `x` |
    /// | `x4` | `y` |
    /// | `x5`, `x6` | `m`, `n` |
    /// | `x7` | alpha bits |
    /// | `x8` / `w8` | group `g`, the ZA vector-group select |
    /// | `x9` | `r0`, first row of the row block |
    /// | `x10` | `r0 + 32g`, first row of group `g` |
    /// | `x12` | `G`, groups of the row block |
    /// | `x13` | `j0`, first column of the column pair |
    /// | `x14`, `x20` | `A(0, j0)`, `A(0, j0 + 1)` |
    /// | `x16` | `j0 + 2` |
    /// | `z0`–`z3`, `z4`–`z7` | rows of group `g` of columns `j0`, `j0 + 1` (and `y` through ZA) |
    /// | `z8` | `alpha · x[j0..j0 + 2]`, replicated per 128-bit segment |
    /// | `z28` | alpha broadcast |
    /// | `p1` | columns `< n` of the column pair (for `ld1rqd`) |
    /// | `pn9` | rows `< m` of group `g` (32-lane counter) |
    ///
    /// It reads `A(i, j)` and `y[i]` only for `i < m`, `j < n`, and `x[j]`
    /// only for `j < n`; it writes only `y[0..m]`. It never writes the
    /// argument block and leaves through the frame's exit label.
    ///
    /// # Safety
    ///
    /// As for `sme2_gemv_notrans_f32`, and the CPU must have
    /// FEAT_SME_F64F64 (`detect::enabled::<f64>()`).
    pub(super) streaming fn sme2_gemv_notrans_f64(args: &GemvArgs<f64>) {
        extensions: [f64f64],
        consts: [
            off_a = OFF_GV_A,
            off_lda = OFF_GV_LDA,
            off_x = OFF_GV_X,
            off_out = OFF_GV_OUT,
            off_m = OFF_GV_M,
            off_n = OFF_GV_N,
            off_alpha = OFF_GV_ALPHA,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_lda}]",
            "ldr x3, [x0, #{off_x}]",
            "ldr x4, [x0, #{off_out}]",
            "ldr x5, [x0, #{off_m}]",
            "ldr x6, [x0, #{off_n}]",
            "ldr x7, [x0, #{off_alpha}]",
            "lsl x2, x2, #3",
        ],
        body: [
            "dup z28.d, x7",
            "mov x9, #0",
            "20:",
            "mov x8, #0",
            "mov x10, x9",
            "21:",
            "whilelt pn9.d, x10, x5, vlx4",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x4, x10, lsl #3]",
            "mova za.d[w8, 0, vgx4], {{z0.d-z3.d}}",
            "add x8, x8, #1",
            "add x10, x10, #32",
            "cmp x8, #16",
            "b.hs 22f",
            "cmp x10, x5",
            "b.lo 21b",
            "22:",
            "mov x12, x8",
            "mov x13, #0",
            "mov x14, x1",
            "30:",
            "add x16, x13, #2",
            "cmp x16, x6",
            "b.hi 40f",
            "whilelt p1.d, x13, x6",
            "ld1rqd {{z8.d}}, p1/z, [x3, x13, lsl #3]",
            "fmul z8.d, z8.d, z28.d",
            "add x20, x14, x2",
            "mov x8, #0",
            "mov x10, x9",
            "31:",
            "whilelt pn9.d, x10, x5, vlx4",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14, x10, lsl #3]",
            "fmla za.d[w8, 0, vgx4], {{z0.d-z3.d}}, z8.d[0]",
            "ld1d {{z4.d-z7.d}}, pn9/z, [x20, x10, lsl #3]",
            "fmla za.d[w8, 0, vgx4], {{z4.d-z7.d}}, z8.d[1]",
            "add x8, x8, #1",
            "add x10, x10, #32",
            "cmp x8, x12",
            "b.lo 31b",
            "mov x13, x16",
            "add x14, x14, x2, lsl #1",
            "b 30b",
            "40:",
            "cmp x13, x6",
            "b.hs 50f",
            "whilelt p1.d, x13, x6",
            "ld1rqd {{z8.d}}, p1/z, [x3, x13, lsl #3]",
            "fmul z8.d, z8.d, z28.d",
            "mov x8, #0",
            "mov x10, x9",
            "41:",
            "whilelt pn9.d, x10, x5, vlx4",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14, x10, lsl #3]",
            "fmla za.d[w8, 0, vgx4], {{z0.d-z3.d}}, z8.d[0]",
            "add x8, x8, #1",
            "add x10, x10, #32",
            "cmp x8, x12",
            "b.lo 41b",
            "50:",
            "mov x8, #0",
            "mov x10, x9",
            "51:",
            "whilelt pn9.d, x10, x5, vlx4",
            "mova {{z0.d-z3.d}}, za.d[w8, 0, vgx4]",
            "st1d {{z0.d-z3.d}}, pn9, [x4, x10, lsl #3]",
            "add x8, x8, #1",
            "add x10, x10, #32",
            "cmp x8, x12",
            "b.lo 51b",
            "add x9, x9, #512",
            "cmp x9, x5",
            "b.lo 20b",
        ],
    }

    /// The 16 lane partials of every column of an `f32` column-major A
    /// against `x[0..m]`, stored to `out[16·j .. 16·j + 16]` for `j < n`, in
    /// one streaming region (module docs: "Trans"; [`trans_finish`] turns
    /// them into the dot products).
    ///
    /// Lane `l` of column `j` is `fma(A(i, j), x[i], ·)` from `+0` over the
    /// rows `i = i0 + l` of every 16-row step below `body = m - m % 16`, in
    /// ascending order, then over the rows of the vector tail that
    /// [`vector_tail`] assigns to it (base `tb`, lanes `tl..th`).
    ///
    /// Loop nest (labels in brackets):
    ///
    /// ```text
    /// r = m % 16, body = m - r; (tb, tl, th) = vector tail   [22] [23]
    /// p6 = lanes tl..th
    /// for jb in (0..n).step_by(64)                       [20] column block
    ///     zero za; nb = min(64, n - jb)
    ///     for g in 0..ceil(nb / 4)                       [30] column group
    ///         lim_k = (4g + k < nb) ? m : 0, k = 0..3
    ///         while i + 64 <= body                       [41] 64-row steps
    ///             x rows i..i+64 -> z8..z11; column k -> z16+k, z20+k, z24+k, z28+k
    ///             za[g] += {z16-z19} · z8, {z20-z23} · z9, {z24-z27} · z10, {z28-z31} · z11
    ///         while i + 16 <= body                       [42] 16-row steps
    ///             za[g] += {z16-z19} · z8 (one row step)
    ///         if th != 0                                 [43] vector tail
    ///             za[g] += {z16-z19} · z8 at base tb, lanes p6,
    ///             the other lanes of z8 set to -0 (they add -0)
    ///                                                    [49] next group
    ///     for g: ZA group g -> out, elements < 16 n      [50]
    /// ```
    ///
    /// Register map:
    ///
    /// | Register | Use |
    /// |---|---|
    /// | `x1` | `a`, `A(0, 0)` |
    /// | `x2` | `lda` in bytes |
    /// | `x3` | `x` |
    /// | `x4` | `out`, the lane partials |
    /// | `x5`, `x6` | `m`, `n` |
    /// | `x7` | `r = m % 16`; from label 23 on `16 n`, the partials to store |
    /// | `x8` / `w8` | group `g`, the ZA vector-group select |
    /// | `x9` | `jb`, first column of the column block |
    /// | `x10` | `nb`, live columns of the block |
    /// | `x11`, `x24`, `x25`, `x26` | `A(0, jb + 4g + k)` for `k` = 0..3 |
    /// | `x12`–`x15` | `lim_k`: `m` for a live column `k`, else 0 |
    /// | `x16` | scratch: column index, next row |
    /// | `x20` | `body` |
    /// | `x21`, `x22`, `x23` | `tb`, `tl`, `th`; `x22` is the partial index at label 50 |
    /// | `x24` | 0 while the tail predicate is built |
    /// | `x27` | `i`, first row of the row step |
    /// | `z8`–`z11` | `x` of the row step |
    /// | `z16`–`z31` | A of the row step, four columns |
    /// | `p0` | all-true |
    /// | `p1`, `p2`–`p5` | rows `< m` (`x`), rows `< lim_k` (column `k`) of a 16-row or tail step |
    /// | `p6`, `p7` | lanes `tl..th` of the tail; lanes `0..tl`, then the dead lanes of the tail's `x` load |
    /// | `pn9`, `pn10`–`pn13` | the same as 64-lane counters (64-row steps; `pn9` also stores) |
    ///
    /// It reads `A(i, j)` and `x[i]` only for `i < m`, `j < n`, and writes
    /// only `out[0..16 n]`. It never writes the argument block and leaves
    /// through the frame's exit label.
    ///
    /// # Safety
    ///
    /// - `detect::thread_ready()` must have returned `true` on this thread.
    /// - `m >= 16`, `n >= 1`; `A(i, j)` at `a + i + j·lda` must be readable
    ///   for `i < m`, `j < n`, `x[0..m]` readable, `out[0..16 n]` writable,
    ///   and `out` must not overlap A, `x` or the argument block.
    pub(super) streaming fn sme2_gemv_trans_f32(args: &GemvArgs<f32>) {
        extensions: [],
        consts: [
            off_a = OFF_GV_A,
            off_lda = OFF_GV_LDA,
            off_x = OFF_GV_X,
            off_out = OFF_GV_OUT,
            off_m = OFF_GV_M,
            off_n = OFF_GV_N,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_lda}]",
            "ldr x3, [x0, #{off_x}]",
            "ldr x4, [x0, #{off_out}]",
            "ldr x5, [x0, #{off_m}]",
            "ldr x6, [x0, #{off_n}]",
            "lsl x2, x2, #2",
        ],
        body: [
            "ptrue p0.s",
            "and x7, x5, #15",
            "sub x20, x5, x7",
            "mov x21, x20",
            "mov x22, #0",
            "mov x23, #0",
            "cmp x7, #8",
            "b.lo 22f",
            "mov x23, #8",
            "cmp x7, #12",
            "b.lo 23f",
            "mov x23, #12",
            "b 23f",
            "22:",
            "cmp x7, #4",
            "b.lo 23f",
            "sub x21, x20, #8",
            "mov x22, #8",
            "mov x23, #12",
            "23:",
            "mov x24, #0",
            "whilelt p6.s, x24, x23",
            "whilelt p7.s, x24, x22",
            "bic p6.b, p0/z, p6.b, p7.b",
            "lsl x7, x6, #4",
            "mov x9, #0",
            "20:",
            "zero {{za}}",
            "sub x10, x6, x9",
            "mov x16, #64",
            "cmp x10, x16",
            "csel x10, x10, x16, lo",
            "mul x11, x9, x2",
            "add x11, x1, x11",
            "mov x8, #0",
            "30:",
            "lsl x16, x8, #2",
            "cmp x16, x10",
            "csel x12, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x13, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x14, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x15, x5, xzr, lo",
            "add x24, x11, x2",
            "add x25, x24, x2",
            "add x26, x25, x2",
            "mov x27, #0",
            "41:",
            "add x16, x27, #64",
            "cmp x16, x20",
            "b.hi 42f",
            "whilelt pn9.s, x27, x5, vlx4",
            "whilelt pn10.s, x27, x12, vlx4",
            "whilelt pn11.s, x27, x13, vlx4",
            "whilelt pn12.s, x27, x14, vlx4",
            "whilelt pn13.s, x27, x15, vlx4",
            "ld1w {{z8.s-z11.s}}, pn9/z, [x3, x27, lsl #2]",
            "ld1w {{z16.s, z20.s, z24.s, z28.s}}, pn10/z, [x11, x27, lsl #2]",
            "ld1w {{z17.s, z21.s, z25.s, z29.s}}, pn11/z, [x24, x27, lsl #2]",
            "ld1w {{z18.s, z22.s, z26.s, z30.s}}, pn12/z, [x25, x27, lsl #2]",
            "ld1w {{z19.s, z23.s, z27.s, z31.s}}, pn13/z, [x26, x27, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z16.s-z19.s}}, z8.s",
            "fmla za.s[w8, 0, vgx4], {{z20.s-z23.s}}, z9.s",
            "fmla za.s[w8, 0, vgx4], {{z24.s-z27.s}}, z10.s",
            "fmla za.s[w8, 0, vgx4], {{z28.s-z31.s}}, z11.s",
            "mov x27, x16",
            "b 41b",
            "42:",
            "add x16, x27, #16",
            "cmp x16, x20",
            "b.hi 43f",
            "whilelt p1.s, x27, x5",
            "whilelt p2.s, x27, x12",
            "whilelt p3.s, x27, x13",
            "whilelt p4.s, x27, x14",
            "whilelt p5.s, x27, x15",
            "ld1w {{z8.s}}, p1/z, [x3, x27, lsl #2]",
            "ld1w {{z16.s}}, p2/z, [x11, x27, lsl #2]",
            "ld1w {{z17.s}}, p3/z, [x24, x27, lsl #2]",
            "ld1w {{z18.s}}, p4/z, [x25, x27, lsl #2]",
            "ld1w {{z19.s}}, p5/z, [x26, x27, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z16.s-z19.s}}, z8.s",
            "mov x27, x16",
            "b 42b",
            "43:",
            "cbz x23, 49f",
            "whilelt p1.s, x21, x5",
            "and p1.b, p6/z, p1.b, p1.b",
            "whilelt p2.s, x21, x12",
            "and p2.b, p6/z, p2.b, p2.b",
            "whilelt p3.s, x21, x13",
            "and p3.b, p6/z, p3.b, p3.b",
            "whilelt p4.s, x21, x14",
            "and p4.b, p6/z, p4.b, p4.b",
            "whilelt p5.s, x21, x15",
            "and p5.b, p6/z, p5.b, p5.b",
            "ld1w {{z8.s}}, p1/z, [x3, x21, lsl #2]",
            "eor p7.b, p0/z, p1.b, p0.b",
            "fneg z8.s, p7/m, z8.s",
            "ld1w {{z16.s}}, p2/z, [x11, x21, lsl #2]",
            "ld1w {{z17.s}}, p3/z, [x24, x21, lsl #2]",
            "ld1w {{z18.s}}, p4/z, [x25, x21, lsl #2]",
            "ld1w {{z19.s}}, p5/z, [x26, x21, lsl #2]",
            "fmla za.s[w8, 0, vgx4], {{z16.s-z19.s}}, z8.s",
            "49:",
            "add x8, x8, #1",
            "add x11, x11, x2, lsl #2",
            "lsl x16, x8, #2",
            "cmp x16, x10",
            "b.lo 30b",
            "mov x8, #0",
            "lsl x22, x9, #4",
            "50:",
            "mova {{z16.d-z19.d}}, za.d[w8, 0, vgx4]",
            "whilelt pn9.s, x22, x7, vlx4",
            "st1w {{z16.s-z19.s}}, pn9, [x4, x22, lsl #2]",
            "add x22, x22, #64",
            "add x8, x8, #1",
            "lsl x16, x8, #2",
            "cmp x16, x10",
            "b.lo 50b",
            "add x9, x9, #64",
            "cmp x9, x6",
            "b.lo 20b",
        ],
    }

    /// The 8 lane partials of every column of an `f64` column-major A
    /// against `x[0..m]`, stored to `out[8·j .. 8·j + 8]` for `j < n`, in
    /// one streaming region (module docs: "Trans"): the `f32` kernel with
    /// 8-lane vectors, so a strided four-vector load covers 32 rows, a row
    /// step 8 rows, and the vector tail takes 4 / 2 rows (lanes `0..4`,
    /// `4..6`, or `4..6` from base `body - 4`).
    ///
    /// Loop nest (labels in brackets):
    ///
    /// ```text
    /// r = m % 8, body = m - r; (tb, tl, th) = vector tail    [22] [23]
    /// p6 = lanes tl..th
    /// for jb in (0..n).step_by(64)                       [20] column block
    ///     zero za; nb = min(64, n - jb)
    ///     for g in 0..ceil(nb / 4)                       [30] column group
    ///         lim_k = (4g + k < nb) ? m : 0, k = 0..3
    ///         while i + 32 <= body                       [41] 32-row steps
    ///         while i + 8 <= body                        [42] 8-row steps
    ///         if th != 0                                 [43] vector tail
    ///                                                    [49] next group
    ///     for g: ZA group g -> out, elements < 8 n       [50]
    /// ```
    ///
    /// Register map: as for `sme2_gemv_trans_f32`, with `x7 = r = m % 8`,
    /// then `8 n`, and 8-lane predicates and counters.
    ///
    /// It reads `A(i, j)` and `x[i]` only for `i < m`, `j < n`, and writes
    /// only `out[0..8 n]`. It never writes the argument block and leaves
    /// through the frame's exit label.
    ///
    /// # Safety
    ///
    /// As for `sme2_gemv_trans_f32` (with `out[0..8 n]`), and the CPU must
    /// have FEAT_SME_F64F64 (`detect::enabled::<f64>()`).
    pub(super) streaming fn sme2_gemv_trans_f64(args: &GemvArgs<f64>) {
        extensions: [f64f64],
        consts: [
            off_a = OFF_GV_A,
            off_lda = OFF_GV_LDA,
            off_x = OFF_GV_X,
            off_out = OFF_GV_OUT,
            off_m = OFF_GV_M,
            off_n = OFF_GV_N,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_lda}]",
            "ldr x3, [x0, #{off_x}]",
            "ldr x4, [x0, #{off_out}]",
            "ldr x5, [x0, #{off_m}]",
            "ldr x6, [x0, #{off_n}]",
            "lsl x2, x2, #3",
        ],
        body: [
            "ptrue p0.d",
            "and x7, x5, #7",
            "sub x20, x5, x7",
            "mov x21, x20",
            "mov x22, #0",
            "mov x23, #0",
            "cmp x7, #4",
            "b.lo 22f",
            "mov x23, #4",
            "cmp x7, #6",
            "b.lo 23f",
            "mov x23, #6",
            "b 23f",
            "22:",
            "cmp x7, #2",
            "b.lo 23f",
            "sub x21, x20, #4",
            "mov x22, #4",
            "mov x23, #6",
            "23:",
            "mov x24, #0",
            "whilelt p6.d, x24, x23",
            "whilelt p7.d, x24, x22",
            "bic p6.b, p0/z, p6.b, p7.b",
            "lsl x7, x6, #3",
            "mov x9, #0",
            "20:",
            "zero {{za}}",
            "sub x10, x6, x9",
            "mov x16, #64",
            "cmp x10, x16",
            "csel x10, x10, x16, lo",
            "mul x11, x9, x2",
            "add x11, x1, x11",
            "mov x8, #0",
            "30:",
            "lsl x16, x8, #2",
            "cmp x16, x10",
            "csel x12, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x13, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x14, x5, xzr, lo",
            "add x16, x16, #1",
            "cmp x16, x10",
            "csel x15, x5, xzr, lo",
            "add x24, x11, x2",
            "add x25, x24, x2",
            "add x26, x25, x2",
            "mov x27, #0",
            "41:",
            "add x16, x27, #32",
            "cmp x16, x20",
            "b.hi 42f",
            "whilelt pn9.d, x27, x5, vlx4",
            "whilelt pn10.d, x27, x12, vlx4",
            "whilelt pn11.d, x27, x13, vlx4",
            "whilelt pn12.d, x27, x14, vlx4",
            "whilelt pn13.d, x27, x15, vlx4",
            "ld1d {{z8.d-z11.d}}, pn9/z, [x3, x27, lsl #3]",
            "ld1d {{z16.d, z20.d, z24.d, z28.d}}, pn10/z, [x11, x27, lsl #3]",
            "ld1d {{z17.d, z21.d, z25.d, z29.d}}, pn11/z, [x24, x27, lsl #3]",
            "ld1d {{z18.d, z22.d, z26.d, z30.d}}, pn12/z, [x25, x27, lsl #3]",
            "ld1d {{z19.d, z23.d, z27.d, z31.d}}, pn13/z, [x26, x27, lsl #3]",
            "fmla za.d[w8, 0, vgx4], {{z16.d-z19.d}}, z8.d",
            "fmla za.d[w8, 0, vgx4], {{z20.d-z23.d}}, z9.d",
            "fmla za.d[w8, 0, vgx4], {{z24.d-z27.d}}, z10.d",
            "fmla za.d[w8, 0, vgx4], {{z28.d-z31.d}}, z11.d",
            "mov x27, x16",
            "b 41b",
            "42:",
            "add x16, x27, #8",
            "cmp x16, x20",
            "b.hi 43f",
            "whilelt p1.d, x27, x5",
            "whilelt p2.d, x27, x12",
            "whilelt p3.d, x27, x13",
            "whilelt p4.d, x27, x14",
            "whilelt p5.d, x27, x15",
            "ld1d {{z8.d}}, p1/z, [x3, x27, lsl #3]",
            "ld1d {{z16.d}}, p2/z, [x11, x27, lsl #3]",
            "ld1d {{z17.d}}, p3/z, [x24, x27, lsl #3]",
            "ld1d {{z18.d}}, p4/z, [x25, x27, lsl #3]",
            "ld1d {{z19.d}}, p5/z, [x26, x27, lsl #3]",
            "fmla za.d[w8, 0, vgx4], {{z16.d-z19.d}}, z8.d",
            "mov x27, x16",
            "b 42b",
            "43:",
            "cbz x23, 49f",
            "whilelt p1.d, x21, x5",
            "and p1.b, p6/z, p1.b, p1.b",
            "whilelt p2.d, x21, x12",
            "and p2.b, p6/z, p2.b, p2.b",
            "whilelt p3.d, x21, x13",
            "and p3.b, p6/z, p3.b, p3.b",
            "whilelt p4.d, x21, x14",
            "and p4.b, p6/z, p4.b, p4.b",
            "whilelt p5.d, x21, x15",
            "and p5.b, p6/z, p5.b, p5.b",
            "ld1d {{z8.d}}, p1/z, [x3, x21, lsl #3]",
            "eor p7.b, p0/z, p1.b, p0.b",
            "fneg z8.d, p7/m, z8.d",
            "ld1d {{z16.d}}, p2/z, [x11, x21, lsl #3]",
            "ld1d {{z17.d}}, p3/z, [x24, x21, lsl #3]",
            "ld1d {{z18.d}}, p4/z, [x25, x21, lsl #3]",
            "ld1d {{z19.d}}, p5/z, [x26, x21, lsl #3]",
            "fmla za.d[w8, 0, vgx4], {{z16.d-z19.d}}, z8.d",
            "49:",
            "add x8, x8, #1",
            "add x11, x11, x2, lsl #2",
            "lsl x16, x8, #2",
            "cmp x16, x10",
            "b.lo 30b",
            "mov x8, #0",
            "lsl x22, x9, #3",
            "50:",
            "mova {{z16.d-z19.d}}, za.d[w8, 0, vgx4]",
            "whilelt pn9.d, x22, x7, vlx4",
            "st1d {{z16.d-z19.d}}, pn9, [x4, x22, lsl #3]",
            "add x22, x22, #32",
            "add x8, x8, #1",
            "lsl x16, x8, #2",
            "cmp x16, x10",
            "b.lo 50b",
            "add x9, x9, #64",
            "cmp x9, x6",
            "b.lo 20b",
        ],
    }
}

/// A GEMV kernel as the self-check and the tests see it.
pub(super) type GemvKernelFn<T> = unsafe fn(&GemvArgs<T>);

thread_local! {
    /// Lane partials of the `f32` `Trans` kernel on this thread.
    static SCRATCH_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// Lane partials of the `f64` `Trans` kernel on this thread.
    static SCRATCH_F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` on `len` elements of the thread-local buffer `key` (grown with
/// zeros on first use, then reused). The buffer is taken out of its cell
/// while `f` runs, so a nested use (impossible today) or a thread that is
/// being torn down gets a fresh buffer instead of a panic.
fn with_buffer<T: SmeElement, R>(
    key: &'static LocalKey<RefCell<Vec<T>>>,
    len: usize,
    f: impl FnOnce(&mut [T]) -> R,
) -> R {
    let mut buffer: Vec<T> = key
        .try_with(|cell| {
            cell.try_borrow_mut()
                .map(|mut held| core::mem::take(&mut *held))
                .unwrap_or_default()
        })
        .unwrap_or_default();
    if buffer.len() < len {
        buffer.resize(len, T::ZERO);
    }
    let result = f(&mut buffer[..len]);
    let _ = key.try_with(move |cell| {
        if let Ok(mut held) = cell.try_borrow_mut()
            && held.capacity() < buffer.capacity()
        {
            *held = buffer;
        }
    });
    result
}

/// The element types of the SME2 GEMV, with their kernels, the NEON
/// kernels they must equal, and the lane reduction of the `Trans` finish.
pub(super) trait SmeGemv: SmeElement + Field + Send + Sync {
    /// Lanes of one streaming vector, the `V` of the module docs: 16
    /// (`f32`) or 8 (`f64`). Also the `NoTrans` part boundary step.
    const LANES: usize;

    /// The `NoTrans` kernel.
    ///
    /// # Safety
    ///
    /// The `# Safety` section of `sme2_gemv_notrans_f32` /
    /// `sme2_gemv_notrans_f64`.
    unsafe fn notrans_kernel(args: &GemvArgs<Self>);

    /// The `Trans` kernel.
    ///
    /// # Safety
    ///
    /// The `# Safety` section of `sme2_gemv_trans_f32` /
    /// `sme2_gemv_trans_f64`.
    unsafe fn trans_kernel(args: &GemvArgs<Self>);

    /// `neon::notrans_f32` / `neon::notrans_f64`.
    fn neon_notrans(alpha: Self, a: MatRef<'_, Self>, x: &[Self], y: &mut [Self]);

    /// `neon::trans_f32` / `neon::trans_f64`.
    fn neon_trans(alpha: Self, a: MatRef<'_, Self>, x: &[Self], y: &mut [Self]);

    /// The pairwise lane reduction of the NEON `Trans` kernel on the
    /// [`LANES`](Self::LANES) partials of one column: `f32`
    /// `v[l] = (L[l] + L[4 + l]) + (L[8 + l] + L[12 + l])`,
    /// `s = (v[0] + v[1]) + (v[2] + v[3])`; `f64`
    /// `v[l] = (L[l] + L[2 + l]) + (L[4 + l] + L[6 + l])`, `s = v[0] + v[1]`.
    fn reduce(lanes: &[Self]) -> Self;

    /// Runs `f` on `len` elements of this thread's scratch buffer.
    fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [Self]) -> R) -> R;
}

impl SmeGemv for f32 {
    const LANES: usize = 16;

    unsafe fn notrans_kernel(args: &GemvArgs<f32>) {
        // SAFETY: the caller upholds this function's contract, which is the
        // kernel's.
        unsafe { sme2_gemv_notrans_f32(args) }
    }

    unsafe fn trans_kernel(args: &GemvArgs<f32>) {
        // SAFETY: as above.
        unsafe { sme2_gemv_trans_f32(args) }
    }

    fn neon_notrans(alpha: f32, a: MatRef<'_, f32>, x: &[f32], y: &mut [f32]) {
        neon::notrans_f32(alpha, a, x, y);
    }

    fn neon_trans(alpha: f32, a: MatRef<'_, f32>, x: &[f32], y: &mut [f32]) {
        neon::trans_f32(alpha, a, x, y);
    }

    fn reduce(lanes: &[f32]) -> f32 {
        let v = |l: usize| (lanes[l] + lanes[4 + l]) + (lanes[8 + l] + lanes[12 + l]);
        (v(0) + v(1)) + (v(2) + v(3))
    }

    fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
        with_buffer(&SCRATCH_F32, len, f)
    }
}

impl SmeGemv for f64 {
    const LANES: usize = 8;

    unsafe fn notrans_kernel(args: &GemvArgs<f64>) {
        // SAFETY: the caller upholds this function's contract, which is the
        // kernel's.
        unsafe { sme2_gemv_notrans_f64(args) }
    }

    unsafe fn trans_kernel(args: &GemvArgs<f64>) {
        // SAFETY: as above.
        unsafe { sme2_gemv_trans_f64(args) }
    }

    fn neon_notrans(alpha: f64, a: MatRef<'_, f64>, x: &[f64], y: &mut [f64]) {
        neon::notrans_f64(alpha, a, x, y);
    }

    fn neon_trans(alpha: f64, a: MatRef<'_, f64>, x: &[f64], y: &mut [f64]) {
        neon::trans_f64(alpha, a, x, y);
    }

    fn reduce(lanes: &[f64]) -> f64 {
        let v = |l: usize| (lanes[l] + lanes[2 + l]) + (lanes[4 + l] + lanes[6 + l]);
        v(0) + v(1)
    }

    fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
        with_buffer(&SCRATCH_F64, len, f)
    }
}

/// The vector tail of a `Trans` column of `m` rows with `lanes` = 16
/// (`f32`) or 8 (`f64`) lanes: `(base, lo, hi)` when one predicated step
/// adds rows `base + lo .. base + hi` to lanes `lo .. hi` (lane `l` takes
/// row `base + l`), `None` when the tail is scalar only. With
/// `r = m % lanes`, `body = m - r` and `h = lanes / 2`, `q = lanes / 4`:
///
/// - `r >= h + q`: `(body, 0, h + q)` (NEON: `h` rows to lanes `0..h`, then
///   `q` rows to lanes `h..h + q`);
/// - `h <= r < h + q`: `(body, 0, h)`;
/// - `q <= r < h`: `(body - h, h, h + q)` (the `q` rows after `body` go to
///   lanes `h..h + q`, loaded from `h` rows below);
/// - `r < q`: `None`.
///
/// The scalar tail then starts at `base + hi` (or `body`).
pub(super) fn vector_tail(m: usize, lanes: usize) -> Option<(usize, usize, usize)> {
    let (h, q) = (lanes / 2, lanes / 4);
    let r = m % lanes;
    let body = m - r;
    if r >= h + q {
        Some((body, 0, h + q))
    } else if r >= h {
        Some((body, 0, h))
    } else if r >= q {
        body.checked_sub(h).map(|base| (base, h, h + q))
    } else {
        None
    }
}

/// First row of the scalar tail of a `Trans` column of `m` rows.
pub(super) fn scalar_start(m: usize, lanes: usize) -> usize {
    match vector_tail(m, lanes) {
        Some((base, _, hi)) => base + hi,
        None => m - m % lanes,
    }
}

/// The dot product of one column from its lane partials, exactly as the
/// NEON `Trans` kernel finishes it: the pairwise reduction
/// ([`SmeGemv::reduce`]), then the scalar tail `s = fma(A[i, j], x[i], s)`
/// for the rows from [`scalar_start`] to `m` in ascending order. `col` is
/// the column (`m` elements), `x` has `m` elements. The caller then forms
/// `y[j] + alpha · s` (a multiplication, then an addition).
pub(super) fn trans_finish<T: SmeGemv>(lanes: &[T], col: &[T], x: &[T]) -> T {
    let m = col.len().min(x.len());
    let start = scalar_start(m, T::LANES).min(m);
    let mut s = T::reduce(lanes);
    for (c, xv) in col[start..m].iter().zip(&x[start..m]) {
        s = SmeElement::fused(*c, *xv, s);
    }
    s
}

/// The crossover of the decision: `m >= `[`MIN_ROWS`]`, m·n >= `[`MIN_WORK`],
/// and for `Trans` (`trans`) also `n >= `[`TRANS_MIN_COLS`].
pub(super) fn crossover(trans: bool, m: usize, n: usize) -> bool {
    m >= MIN_ROWS && m.saturating_mul(n) >= MIN_WORK && (!trans || n >= TRANS_MIN_COLS)
}

/// The number of parts of a call on a matrix of `bytes` bytes: 1 unless
/// `par` is parallel, `units >= 2` and `bytes >= min_bytes`; then
/// `min(units, par.num_threads())`, at least 1. `par.num_threads()` is asked
/// for last (it builds rayon's global pool for `Par::Rayon`).
pub(super) fn part_count(bytes: usize, par: Par, units: usize, min_bytes: usize) -> usize {
    if units < 2 || par.is_sequential() || bytes < min_bytes {
        return 1;
    }
    units.min(par.num_threads()).max(1)
}

/// One part of a call: `len` rows (`NoTrans`) or columns (`Trans`) from
/// `start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Piece {
    /// First row or column.
    pub(super) start: usize,
    /// Rows or columns.
    pub(super) len: usize,
}

/// `len` rows or columns (`len >= 1`) in at most `parts` near-equal runs of
/// whole `unit`s (the last run may end in a partial unit): the part count is
/// clamped to `1..=ceil(len / unit)`, every run is non-empty, every start is
/// a multiple of `unit`, and the runs tile `0..len` in order. Empty for
/// `len == 0` or `unit == 0`.
pub(super) fn split(len: usize, parts: usize, unit: usize) -> Vec<Piece> {
    if len == 0 || unit == 0 {
        return Vec::new();
    }
    let units = len.div_ceil(unit);
    let count = parts.clamp(1, units);
    let point = |index: usize| -> usize {
        // index · units / count <= units, computed without overflow.
        let whole = (index / count) * units;
        let rest = ((index % count) * units) / count;
        (whole + rest).saturating_mul(unit).min(len)
    };
    (0..count)
        .map(|index| {
            let start = point(index);
            Piece {
                start,
                len: point(index + 1) - start,
            }
        })
        .collect()
}

/// The SME2 GEMV: `y += alpha · op(A) · x` on SME2, or a decline.
///
/// Same contract as `gemv_neon_dispatch` / `gemv_neon_parallel_dispatch`
/// in `mod.rs`: `y` is already scaled by beta, `alpha != 0`, `x.len()` and
/// `y.len()` match `op(A)` (`A` is the stored `m × n` matrix). `true`: `y`
/// holds the complete result, bit for bit the NEON kernels' (`neon.rs`)
/// except NaN sign and payload, for every `par`. `false`: `y` is untouched.
/// See the module docs for the decision. `ConjTrans` is `Trans` (real
/// types).
///
/// The size crossover is the first test, before the type hand-off and the
/// detection: a GEMV below it pays a few compares of its dimensions and
/// never starts the detection (or the first-use checks). `T` needs no
/// `Send` or `Sync` (the generic `gemv` has neither): the parallel split
/// works on `f32` / `f64` after the hand-off.
#[inline]
pub(crate) fn try_gemv<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    par: Par,
) -> bool {
    let transposed = !matches!(trans, GemvTrans::NoTrans);
    // 1. The crossover (plain compares of the dimensions).
    if !crossover(transposed, a.nrows(), a.ncols()) {
        return false;
    }
    try_gemv_typed_entry(transposed, alpha, a, x, y, par)
}

/// [`try_gemv`] beyond the crossover: the type hand-off to `f32` / `f64`,
/// then the rest of the decision ([`try_gemv_typed`]).
#[inline(never)]
fn try_gemv_typed_entry<T: Field>(
    transposed: bool,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    par: Par,
) -> bool {
    if let (Some(a32), Some(x32), Some(alpha32)) = (
        same_matref::<T, f32>(a),
        slice_as::<T, f32>(x),
        same_value::<T, f32>(alpha),
    ) && let Some(mut y_view) = same_matmut::<T, f32>(MatMut::from_slice(&mut *y))
        && let Some(y32) = y_view.col_as_slice_mut(0)
    {
        return try_gemv_typed(transposed, alpha32, a32, x32, y32, par);
    }
    if let (Some(a64), Some(x64), Some(alpha64)) = (
        same_matref::<T, f64>(a),
        slice_as::<T, f64>(x),
        same_value::<T, f64>(alpha),
    ) && let Some(mut y_view) = same_matmut::<T, f64>(MatMut::from_slice(&mut *y))
        && let Some(y64) = y_view.col_as_slice_mut(0)
    {
        return try_gemv_typed(transposed, alpha64, a64, x64, y64, par);
    }
    false
}

/// `x` as `&[U]` when `T` is `U`: the matrix-view hand-off of
/// `level1/typed.rs` on `x` as an `len × 1` view. (The slice helpers of that
/// file are compiled only where the NEON kernels are, not under
/// `force-scalar`, which still compiles this module; the view helpers are
/// compiled everywhere.) `y` goes the same way through `same_matmut`.
fn slice_as<T: Scalar, U: Scalar>(x: &[T]) -> Option<&[U]> {
    same_matref::<T, U>(MatRef::from_slice(x)).and_then(|view| view.col_as_slice(0))
}

/// [`try_gemv`] for an SME2 element type, past the crossover: the rest of
/// the decision of the module docs, then the run. `transposed` is `Trans` /
/// `ConjTrans`.
fn try_gemv_typed<T: SmeGemv>(
    transposed: bool,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    par: Par,
) -> bool {
    // 2. Hardware (and the build's caps, and a test's force-off) for `T`.
    //    While another thread runs the first-use check this is `false` at
    //    once (the detection's non-waiting form), and NEON computes the call.
    if !detect::enabled::<T>() {
        return false;
    }
    // 3. Operands.
    let (m, n) = (a.nrows(), a.ncols());
    let (x_len, y_len) = if transposed { (m, n) } else { (n, m) };
    if a.col_stride() != 1 || x.len() != x_len || y.len() != y_len || alpha == T::ZERO {
        return false;
    }
    // 4. The calling thread can enter a streaming region.
    if !detect::thread_ready() {
        return false;
    }
    // 5. Parts.
    let bytes = m
        .saturating_mul(n)
        .saturating_mul(core::mem::size_of::<T>());
    let parts = part_count(bytes, par, detect::units(), par_min_bytes());
    let (len, unit) = if transposed {
        (n, TRANS_SPLIT_COLS)
    } else {
        (m, T::LANES)
    };
    let pieces = split(len, parts, unit);
    if pieces.is_empty() {
        return false;
    }

    #[cfg(test)]
    crate::level3::sme2::test_support::note_decision();
    run(transposed, alpha, a, x, y, &pieces);
    true
}

/// The parallel threshold of the calling thread: [`PAR_MIN_BYTES`], or the
/// test override.
fn par_min_bytes() -> usize {
    #[cfg(test)]
    {
        SWITCHES
            .with(Cell::get)
            .par_min_bytes
            .unwrap_or(PAR_MIN_BYTES)
    }
    #[cfg(not(test))]
    {
        PAR_MIN_BYTES
    }
}

/// Computes every piece of the call: one on the calling thread, more on
/// rayon (one task per piece, on the ambient pool). Each piece checks
/// `detect::thread_ready()` on the thread that runs it and falls back to
/// the NEON kernel on its sub-view when that thread is not ready.
fn run<T: SmeGemv>(
    transposed: bool,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    pieces: &[Piece],
) {
    #[cfg(test)]
    let rendezvous = SWITCHES
        .with(Cell::get)
        .rendezvous
        .then(|| Rendezvous::new(pieces.len()));
    let mut tasks: Vec<(Piece, &mut [T])> = Vec::with_capacity(pieces.len());
    let mut rest = y;
    for piece in pieces {
        let Some((head, tail)) = rest.split_at_mut_checked(piece.len) else {
            break;
        };
        tasks.push((*piece, head));
        rest = tail;
    }
    let (m, n) = (a.nrows(), a.ncols());
    let run_one = |(piece, chunk): (Piece, &mut [T])| {
        #[cfg(test)]
        let (entries, timed_out) = (
            crate::level3::sme2::test_support::kernel_entries_here(),
            rendezvous.as_ref().is_some_and(Rendezvous::arrive),
        );
        let view = if transposed {
            a.submatrix(0, piece.start, m, piece.len)
        } else {
            a.submatrix(piece.start, 0, piece.len, n)
        };
        let sme2 = detect::thread_ready();
        match (sme2, transposed) {
            (true, false) => sme2_notrans_part(alpha, view, x, chunk),
            (true, true) => sme2_trans_part(alpha, view, x, chunk),
            (false, false) => T::neon_notrans(alpha, view, x, chunk),
            (false, true) => T::neon_trans(alpha, view, x, chunk),
        }
        #[cfg(test)]
        let done = PartRecord::here(piece, sme2, entries, timed_out);
        #[cfg(not(test))]
        let done = sme2;
        done
    };
    #[cfg(feature = "parallel")]
    let records: Vec<_> = if tasks.len() > 1 {
        tasks.into_par_iter().with_max_len(1).map(run_one).collect()
    } else {
        tasks.into_iter().map(run_one).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let records: Vec<_> = tasks.into_iter().map(run_one).collect();
    #[cfg(test)]
    LAST_RUN.with(|last| {
        *last.borrow_mut() = Some(GemvRun {
            transposed,
            parts: records,
        });
    });
    #[cfg(not(test))]
    let _ = records;
}

/// One `NoTrans` part on SME2: one kernel call for the whole sub-view.
/// The calling thread was found ready just before.
fn sme2_notrans_part<T: SmeGemv>(alpha: T, a: MatRef<'_, T>, x: &[T], y: &mut [T]) {
    let (m, n) = (a.nrows(), a.ncols());
    if m == 0 || n == 0 || x.len() != n || y.len() != m || a.col_stride() != 1 {
        T::neon_notrans(alpha, a, x, y);
        return;
    }
    let args = GemvArgs {
        a: a.as_ptr(),
        lda: a.row_stride(),
        x: x.as_ptr(),
        out: y.as_mut_ptr(),
        m,
        n,
        alpha_bits: alpha.bits(),
    };
    // SAFETY: `detect::thread_ready()` returned `true` on this thread just
    // before (`run`), and `detect::enabled::<T>()` held for the call (so
    // FEAT_SME_F64F64 for `f64`). `m, n >= 1`; with `col_stride() == 1`
    // element `(i, j)` of the view is at `a + i + j·lda` for `i < m`,
    // `j < n`; `x` has `n` and `y` has `m` elements, and `y` is an exclusive
    // borrow, so it overlaps neither A, `x` nor `args`.
    unsafe { T::notrans_kernel(&args) };
}

/// One `Trans` part on SME2: one kernel call per at most
/// [`TRANS_COLS_PER_CALL`] columns into the thread's scratch buffer, then
/// [`trans_finish`] and `y[j] + alpha · s` for each of those columns. The
/// calling thread was found ready just before.
fn sme2_trans_part<T: SmeGemv>(alpha: T, a: MatRef<'_, T>, x: &[T], y: &mut [T]) {
    let (m, n) = (a.nrows(), a.ncols());
    if m < MIN_ROWS || n == 0 || x.len() != m || y.len() != n || a.col_stride() != 1 {
        T::neon_trans(alpha, a, x, y);
        return;
    }
    let lanes = T::LANES;
    T::with_scratch(TRANS_COLS_PER_CALL * lanes, |scratch| {
        for (block, ys) in y.chunks_mut(TRANS_COLS_PER_CALL).enumerate() {
            let c0 = block * TRANS_COLS_PER_CALL;
            let cols = ys.len();
            let sub = a.submatrix(0, c0, m, cols);
            let args = GemvArgs {
                a: sub.as_ptr(),
                lda: sub.row_stride(),
                x: x.as_ptr(),
                out: scratch.as_mut_ptr(),
                m,
                n: cols,
                alpha_bits: 0,
            };
            // SAFETY: the thread is ready and `detect::enabled::<T>()` held
            // (see `sme2_notrans_part`); `m >= MIN_ROWS >= 16`,
            // `1 <= cols <= TRANS_COLS_PER_CALL`; element `(i, j)` of `sub`
            // is at `a + i + j·lda` for `i < m`, `j < cols`; `x` has `m`
            // elements; the scratch holds `TRANS_COLS_PER_CALL · lanes >=
            // cols · lanes` elements and is this thread's own buffer, so it
            // overlaps neither A, `x` nor `args`.
            unsafe { T::trans_kernel(&args) };
            for (k, (yv, partials)) in ys.iter_mut().zip(scratch.chunks_exact(lanes)).enumerate() {
                match sub.col_as_slice(k) {
                    Some(col) => {
                        let s = trans_finish(partials, col, x);
                        *yv = *yv + alpha * s;
                    }
                    None => {
                        T::neon_trans(
                            alpha,
                            sub.submatrix(0, k, m, 1),
                            x,
                            core::slice::from_mut(yv),
                        );
                    }
                }
            }
        }
    });
}

/// Runs the GEMV kernels of the present types against plain Rust on a few
/// fixed shapes and returns `true` when all of them give the bits of the
/// definition (`NoTrans`: `y`; `Trans`: every lane partial, `x` and A
/// untouched, nothing written past the partials). One of the first-use
/// checks that `detect::hw()` runs before it reports SME2 hardware (on the
/// first call from a ready thread, with FPCR = 0 and the caller's FPCR and
/// FPSR restored afterwards): `f32` always, `f64` when `f64_present`. Never
/// panics. It calls nothing that runs or reads the detection, and creates
/// no thread.
///
/// Cost: one kernel call per shape of [`CHECK_NOTRANS`] and of
/// [`CHECK_TRANS`] and type, 12 for `f32` and 12 more for `f64`.
///
/// # Safety
///
/// The CPU has SME2 with a 512-bit streaming vector length (and
/// FEAT_SME_F64F64 when `f64_present`), and the calling thread was just
/// found in the state that `detect::thread_ready()` checks.
pub(crate) unsafe fn self_check(f64_present: bool) -> bool {
    let f64_kernels: Option<(GemvKernelFn<f64>, GemvKernelFn<f64>)> = if f64_present {
        Some((sme2_gemv_notrans_f64, sme2_gemv_trans_f64))
    } else {
        None
    };
    // SAFETY: forwarded from this function's contract.
    unsafe { self_check_with((sme2_gemv_notrans_f32, sme2_gemv_trans_f32), f64_kernels) }
}

/// [`self_check`] for given kernels (`(notrans, trans)` per type): `true`
/// only when every given kernel passes every shape of [`CHECK_NOTRANS`] /
/// [`CHECK_TRANS`].
///
/// # Safety
///
/// As for the kernels: each given kernel may be called on blocks that
/// satisfy its documented contract.
pub(super) unsafe fn self_check_with(
    f32_kernels: (GemvKernelFn<f32>, GemvKernelFn<f32>),
    f64_kernels: Option<(GemvKernelFn<f64>, GemvKernelFn<f64>)>,
) -> bool {
    // SAFETY: forwarded from this function's contract.
    let ok32 = unsafe { check_type(f32_kernels.0, f32_kernels.1) };
    ok32 && f64_kernels.is_none_or(|(notrans, trans)| {
        // SAFETY: as above.
        unsafe { check_type(notrans, trans) }
    })
}

/// `NoTrans` shapes `(m, n)` of the self-check: one and several row blocks
/// (a partial last group, a full block of 16 groups for both types), every
/// column remainder, a single element.
pub(super) const CHECK_NOTRANS: [(usize, usize); 5] =
    [(1, 1), (67, 8), (130, 6), (1030, 7), (513, 3)];

/// `Trans` shapes `(m, n)` of the self-check: every vector-tail case of
/// both types (`m % 16` in {0, 2, 5, 10, 14} for `f32`, `m % 8` in
/// {0, 2, 5, 2, 6} for `f64`: every case of [`vector_tail`] for both),
/// 64-row and single row steps, one and several column groups and blocks,
/// a partial last group. The shapes from [`CHECK_NEGATIVE_ZERO_FROM`] on
/// hold the negative-zero data of [`negative_zero_value`] instead of the
/// check values: `m = 70` (`f32`: tail lanes `8..12`, dead lanes `0..8` and
/// `12..16`; `f64`: lanes `0..6`, dead `6..8`) and `m = 76` (`f32`: lanes
/// `0..12`, dead `12..16`; `f64`: lanes `0..4`, dead `4..8`).
pub(super) const CHECK_TRANS: [(usize, usize); 7] = [
    (64, 1),
    (82, 5),
    (101, 67),
    (90, 130),
    (158, 4),
    (70, 3),
    (76, 2),
];

/// The first [`CHECK_TRANS`] shape with the negative-zero data.
pub(super) const CHECK_NEGATIVE_ZERO_FROM: usize = 5;

/// The negative-zero data of a `Trans` shape: every element of A is `-u`
/// and every element of `x` is `u`, with `u = 2^-100` (`f32`) or `2^-600`
/// (`f64`), so every product `-u · u` underflows and, with FPCR = 0 (round
/// to nearest, no flush-to-zero), every lane partial `fma(-u, u, acc)` from
/// `acc = +0` is `-0`. The dead lanes of the vector tail must keep that
/// `-0` (the kernel adds `+0 · -0` there; adding `+0 · +0` would turn it
/// into `+0`), so the check fails for a kernel that touches them. `a_side`
/// selects A's value.
fn negative_zero_value<T: SmeElement>(a_side: bool) -> T {
    let (u, sign) = if core::mem::size_of::<T>() == 4 {
        (0x0d80_0000, 1 << 31)
    } else {
        (0x1a70_0000_0000_0000, 1 << 63)
    };
    T::from_bits64(if a_side { u | sign } else { u })
}

/// Guard elements on each side of every self-check buffer.
const GUARD: usize = 8;

/// A signalling NaN with a recognisable payload.
fn guard_value<T: SmeElement>() -> T {
    if core::mem::size_of::<T>() == 4 {
        T::from_bits64(0x7fa0_0bad)
    } else {
        T::from_bits64(0x7ff4_0000_0000_0bad)
    }
}

/// A deterministic value for element `index` of a buffer: full-mantissa
/// values of mixed sign and magnitude (`2^-8 .. 2^8`), so that sums depend
/// on their order, and some exact zeros, negative zeros and subnormals.
fn check_value<T: SmeElement>(index: usize, salt: u64) -> T {
    // splitmix64 of the index and the salt.
    let mut z = (index as u64)
        .wrapping_add(salt.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let wide = core::mem::size_of::<T>() == 8;
    let (sign_bit, mantissa_bits, bias) = if wide { (63, 52, 1023) } else { (31, 23, 127) };
    let sign = (z & 1) << sign_bit;
    match (z >> 58) % 29 {
        0 => T::ZERO,
        1 => T::from_bits64(1 << sign_bit),
        2 => T::from_bits64(sign | ((z >> 8) & 0xffff) | 1),
        _ => {
            let exponent = bias - 8 + ((z >> 1) & 15);
            let mantissa = (z >> 5) & ((1 << mantissa_bits) - 1);
            T::from_bits64(sign | (exponent << mantissa_bits) | mantissa)
        }
    }
}

/// A buffer of `len` check values framed by [`GUARD`] signalling NaNs.
fn guarded<T: SmeElement>(len: usize, salt: u64) -> Vec<T> {
    let mut buffer = vec![guard_value::<T>(); len + 2 * GUARD];
    for (index, value) in buffer[GUARD..GUARD + len].iter_mut().enumerate() {
        *value = check_value(index, salt);
    }
    buffer
}

/// Bitwise equality, or both NaN.
fn same_bits<T: SmeElement>(got: T, want: T) -> bool {
    got.bits() == want.bits() || (got.nan() && want.nan())
}

/// Whether two buffers hold exactly the same bits.
fn identical<T: SmeElement>(got: &[T], want: &[T]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| g.bits() == w.bits())
}

/// The lane partials of the definition (module docs, "Trans") for column
/// `col` against `x`: `lanes` chains from `+0`.
pub(super) fn reference_lanes<T: SmeElement>(col: &[T], x: &[T], lanes: usize) -> Vec<T> {
    let m = col.len().min(x.len());
    let mut acc = vec![T::ZERO; lanes];
    let body = m - m % lanes;
    for i0 in (0..body).step_by(lanes) {
        for (l, lane) in acc.iter_mut().enumerate() {
            *lane = SmeElement::fused(col[i0 + l], x[i0 + l], *lane);
        }
    }
    if let Some((base, lo, hi)) = vector_tail(m, lanes) {
        for (l, lane) in acc.iter_mut().enumerate().take(hi).skip(lo) {
            *lane = SmeElement::fused(col[base + l], x[base + l], *lane);
        }
    }
    acc
}

/// The self-check of one type's two kernels.
///
/// # Safety
///
/// As for [`self_check_with`].
unsafe fn check_type<T: SmeGemv>(notrans: GemvKernelFn<T>, trans: GemvKernelFn<T>) -> bool {
    let lanes = T::LANES;
    let alpha = check_value::<T>(5, 77);
    let alpha = if alpha == T::ZERO || alpha.nan() {
        T::ONE
    } else {
        alpha
    };
    for (case, &(m, n)) in CHECK_NOTRANS.iter().enumerate() {
        let salt = case as u64;
        let lda = m + 3;
        let a = guarded::<T>(lda * n, salt);
        let x = guarded::<T>(n, salt + 100);
        let y0 = guarded::<T>(m, salt + 200);
        let (a_before, x_before) = (a.clone(), x.clone());
        let mut want = y0.clone();
        for i in 0..m {
            let mut acc = want[GUARD + i];
            for j in 0..n {
                let t = alpha * x[GUARD + j];
                acc = SmeElement::fused(a[GUARD + i + j * lda], t, acc);
            }
            want[GUARD + i] = acc;
        }
        let mut got = y0;
        let args = GemvArgs {
            a: a[GUARD..].as_ptr(),
            lda,
            x: x[GUARD..].as_ptr(),
            out: got[GUARD..].as_mut_ptr(),
            m,
            n,
            alpha_bits: alpha.bits(),
        };
        // SAFETY: the caller guarantees the kernel may run on this thread;
        // A holds `lda · n` elements from `a`, `x` has `n` and `y` has `m`
        // elements, all inside their buffers, and `got` is a separate
        // buffer.
        unsafe { notrans(&args) };
        let values_ok = got.iter().zip(&want).enumerate().all(|(index, (g, w))| {
            if (GUARD..GUARD + m).contains(&index) {
                same_bits(*g, *w)
            } else {
                g.bits() == w.bits()
            }
        });
        if !values_ok || !identical(&a, &a_before) || !identical(&x, &x_before) {
            return false;
        }
    }
    for (case, &(m, n)) in CHECK_TRANS.iter().enumerate() {
        let salt = case as u64 + 1000;
        let lda = m + 5;
        let mut a = guarded::<T>(lda * n, salt);
        let mut x = guarded::<T>(m, salt + 100);
        let negative_zero = case >= CHECK_NEGATIVE_ZERO_FROM;
        if negative_zero {
            for j in 0..n {
                a[GUARD + j * lda..GUARD + j * lda + m].fill(negative_zero_value(true));
            }
            x[GUARD..GUARD + m].fill(negative_zero_value(false));
        }
        let (a_before, x_before) = (a.clone(), x.clone());
        let mut got = vec![guard_value::<T>(); n * lanes + 2 * GUARD];
        let args = GemvArgs {
            a: a[GUARD..].as_ptr(),
            lda,
            x: x[GUARD..].as_ptr(),
            out: got[GUARD..].as_mut_ptr(),
            m,
            n,
            alpha_bits: 0,
        };
        // SAFETY: as above; `m >= 16`, the partials buffer holds `n · lanes`
        // elements after the guard.
        unsafe { trans(&args) };
        let guards_ok = got[..GUARD]
            .iter()
            .chain(&got[GUARD + n * lanes..])
            .all(|g| g.bits() == guard_value::<T>().bits());
        let lanes_ok = (0..n).all(|j| {
            let col = &a[GUARD + j * lda..GUARD + j * lda + m];
            let want = reference_lanes(col, &x[GUARD..GUARD + m], lanes);
            // The negative-zero shapes need every reference partial to be
            // `-0` (FPCR = 0 makes it so); otherwise the case would prove
            // nothing, and the check fails.
            let negative_zero_bits = 1 << (8 * core::mem::size_of::<T>() - 1);
            let reference_ok =
                !negative_zero || want.iter().all(|w| w.bits() == negative_zero_bits);
            reference_ok
                && got[GUARD + j * lanes..GUARD + (j + 1) * lanes]
                    .iter()
                    .zip(&want)
                    .all(|(g, w)| same_bits(*g, *w))
        });
        if !guards_ok || !lanes_ok || !identical(&a, &a_before) || !identical(&x, &x_before) {
            return false;
        }
    }
    true
}

/// How one part of a call ran (test builds only).
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PartRecord {
    /// The rows (`NoTrans`) or columns (`Trans`) of the part.
    pub(super) piece: Piece,
    /// The part ran on SME2 (its thread was ready); otherwise on NEON.
    pub(super) sme2: bool,
    /// Kernel entries counted on the part's thread over its run.
    pub(super) kernel_calls: u64,
    /// The thread that ran the part.
    pub(super) thread: ThreadId,
    /// `rayon::current_thread_index()` of that thread (`None` outside
    /// rayon).
    pub(super) worker: Option<usize>,
    /// The part gave up waiting at the rendezvous.
    pub(super) timed_out: bool,
}

#[cfg(test)]
impl PartRecord {
    /// The record of `piece`, which has just run on the calling thread;
    /// `entries` was the thread's kernel-entry count when it started.
    fn here(piece: Piece, sme2: bool, entries: u64, timed_out: bool) -> Self {
        #[cfg(feature = "parallel")]
        let worker = rayon::current_thread_index();
        #[cfg(not(feature = "parallel"))]
        let worker = None;
        Self {
            piece,
            sme2,
            kernel_calls: crate::level3::sme2::test_support::kernel_entries_here()
                .wrapping_sub(entries),
            thread: std::thread::current().id(),
            worker,
            timed_out,
        }
    }
}

/// What one accepted call did (test builds only).
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GemvRun {
    /// `Trans` / `ConjTrans`.
    pub(super) transposed: bool,
    /// The parts, in order.
    pub(super) parts: Vec<PartRecord>,
}

/// Test switches of the calls made on a thread (test builds only).
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Switches {
    /// Replaces [`PAR_MIN_BYTES`].
    pub(super) par_min_bytes: Option<usize>,
    /// Every part waits, before anything else (before its readiness
    /// check), until all parts of the call have arrived, so that the parts
    /// run on distinct threads. The pool needs at least as many threads as
    /// the call has parts; otherwise the parts time out (after
    /// [`RENDEZVOUS_TIMEOUT`], recorded) and carry on.
    pub(super) rendezvous: bool,
}

#[cfg(test)]
thread_local! {
    /// The switches of this thread.
    static SWITCHES: Cell<Switches> = const {
        Cell::new(Switches { par_min_bytes: None, rendezvous: false })
    };
    /// What the last accepted call on this thread did.
    static LAST_RUN: RefCell<Option<GemvRun>> = const { RefCell::new(None) };
}

/// Restores the previous switches of its thread when dropped (test builds
/// only). Neither `Send` nor `Sync`.
#[cfg(test)]
#[derive(Debug)]
#[must_use = "the switches hold only while the guard lives"]
pub(super) struct SwitchGuard {
    /// The switches to restore.
    previous: Switches,
    /// Keeps the guard on its thread.
    _not_send: PhantomData<*const ()>,
}

/// Sets the switches of the calls made on this thread while the guard
/// lives (test builds only). Read once per call, on the calling thread.
#[cfg(test)]
pub(super) fn override_switches(switches: Switches) -> SwitchGuard {
    let previous = SWITCHES.with(|cell| cell.replace(switches));
    SwitchGuard {
        previous,
        _not_send: PhantomData,
    }
}

#[cfg(test)]
impl Drop for SwitchGuard {
    fn drop(&mut self) {
        SWITCHES.with(|cell| cell.set(self.previous));
    }
}

/// The report of the last accepted call on this thread, taken out (test
/// builds only).
#[cfg(test)]
pub(super) fn take_last_run() -> Option<GemvRun> {
    LAST_RUN.with(|last| last.borrow_mut().take())
}

/// How long a part waits at the rendezvous (test builds only).
#[cfg(test)]
pub(super) const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(60);

/// The meeting point of the parts of one call (test builds only).
#[cfg(test)]
#[derive(Debug)]
struct Rendezvous {
    /// The number of parts.
    parts: usize,
    /// Parts arrived so far, and whether a part gave up waiting.
    state: Mutex<(usize, bool)>,
    /// Signalled when the last part arrives or a part gives up.
    arrived: Condvar,
}

#[cfg(test)]
impl Rendezvous {
    /// A rendezvous of `parts` parts.
    fn new(parts: usize) -> Self {
        Self {
            parts,
            state: Mutex::new((0, false)),
            arrived: Condvar::new(),
        }
    }

    /// Arrives and waits until every part has arrived. `true` when the part
    /// stopped waiting before that (it waited [`RENDEZVOUS_TIMEOUT`], or an
    /// earlier part had given up, and then it does not wait).
    fn arrive(&self) -> bool {
        let deadline = Instant::now() + RENDEZVOUS_TIMEOUT;
        let Ok(mut state) = self.state.lock() else {
            return true;
        };
        state.0 += 1;
        if state.0 >= self.parts {
            self.arrived.notify_all();
            return false;
        }
        while state.0 < self.parts && !state.1 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                state.1 = true;
                self.arrived.notify_all();
                break;
            }
            state = match self.arrived.wait_timeout(state, left) {
                Ok((state, _)) => state,
                Err(_) => return true,
            };
        }
        state.0 < self.parts
    }
}
