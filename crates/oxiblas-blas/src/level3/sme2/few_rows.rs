//! SME2 GEMM for products with few rows (`m ≤ 48` for `f32`, `m ≤ 32` for
//! `f64`): [`few_rows_gemm`], one streaming region for the whole product.
//!
//! With few rows every element of B takes part in only `2m` floating-point
//! operations, so writing a packed copy of B and reading it back costs about
//! as much as the arithmetic. The few-rows kernels compute the whole product
//! in one streaming region instead: C lives in the ZA tiles that the few
//! rows need, a free tile transposes B on the fly, A is read in place, and
//! nothing is written but C. At every boundary of the caller's KC partition
//! the tiles are flushed to C and zeroed again, so the result is bit for bit
//! the one of the blocked path.
//!
//! # Shape of the computation
//!
//! - **C in ZA.** `f32`: the rows of C live in the `.s` tiles `za0`–`za2`
//!   (16 rows each, so 48 rows at most), 16 columns per column panel. `f64`:
//!   the `.d` tiles `za0`–`za3` (8 rows each, 32 rows at most), 8 columns per
//!   panel. FMOPA takes the B row as `Zn` and the A vector as `Zm`, so
//!   horizontal slice `c` of a C tile is a contiguous piece of column
//!   `j0 + c` of C, exactly as in the macro-kernels. A tile without a live
//!   row (`m ≤ 16t` / `8t`) is neither zeroed, accumulated nor stored: one
//!   depth-loop variant per number of live tiles.
//! - **op(B) = B (NoTrans, column-major B).** A depth chunk of a column of B
//!   is contiguous, a row of op(B) is not. A free tile transposes: `za3.s`
//!   (`f32`) / `za4.d` (`f64`) takes column `j0 + c` of the panel, depth
//!   steps `q0 .. q0 + 16` (`8`), as its horizontal slice `c`; its vertical
//!   slice `q` is then the B row of depth step `q0 + q` across the panel, the
//!   `Zn` operand of FMOPA (`mova`, four slices at a time for a whole chunk).
//!   The column loads are predicated to the live depth steps of the chunk
//!   (`p5 = whilelt(0, steps left)`), and only the live columns of the panel
//!   are loaded: in-place B has no padding, nothing beyond `op(B)[0..k, cols]`
//!   is read. Slices of the transposer that belong to dead columns keep stale
//!   values; they feed only C slices that are never stored.
//! - **op(B) = B^T (Trans).** A row of op(B) is contiguous in the stored
//!   matrix: it is loaded directly, predicated to the live columns
//!   (`p6 = whilelt(j0, n)`), with no transposition tile.
//! - **A in place.** `A(i, p) = a[i + p · lda]`; every A load is predicated to
//!   the rows below `m` (`p1`, or the counter `pn9` for 2 to 4 tiles), so no
//!   A row `≥ m` is read.
//! - **Exact depth.** Each depth block runs exactly `d = min(kc, k - p0)`
//!   FMOPA steps per tile, in ascending order: whole chunks, then one step at
//!   a time for the last partial chunk (NoTrans), or two steps at a time and
//!   one more for an odd depth (Trans). No step is ever run on padding, so a
//!   padded lane can never turn `0 · ∞` into a NaN of a live element.
//!
//! # Loop order
//!
//! Depth blocks are the outermost loop, column panels the next: for each
//! block `b` (`p0 = b · kc`, `d = min(kc, k - p0)`) the kernel sweeps all
//! column panels, and for each panel zeroes the live C tiles, runs the `d`
//! depth steps and applies the block's epilogue. The `m × d` block of A (at
//! most 48 × kc `f32` or 32 × kc `f64`) is reused by every column panel and
//! stays in the L1/L2 cache, while every element of B is read exactly once
//! whichever loop is outermost. C is read and written once per depth block
//! in either order, so this order costs nothing and keeps A hot.
//!
//! # The value of every C element (bitwise; rules R1–R4 of `args.rs`)
//!
//! ```text
//! for b = 0, 1, …: p0 = b · kc, d = min(kc, k - p0)
//!     acc = +0;  for p in p0 .. p0 + d (ascending): acc = fma(A(i, p), B(p, j), acc)
//!     C(i, j) = epilogue(mode_b, acc, C(i, j))      mode_0 = Epilogue::select(alpha, beta), beta_0 = beta
//!                                                   mode_b = Epilogue::select(alpha, 1),    beta_b = 1   (b ≥ 1)
//! ```
//!
//! That is exactly the sequence of macro-kernel calls of the blocked driver
//! (`driver::run_part`) for the same `kc`, so the result equals the NEON
//! blocked path bit for bit except NaN payloads. The epilogues are those of
//! the macro-kernels (M0 direct slice stores; M1 `acc + C`; M2 `acc · alpha`;
//! M3 `acc · alpha` then `fma(C, beta, t)`), with the mode of the first and
//! the mode of the later depth blocks chosen in Rust ([`Epilogue::select`])
//! and the bit patterns of alpha, beta and one in the argument block.
//!
//! # Floating-point flags
//!
//! FMOPA and the MOVA / slice moves raise no flags; the predicated SVE
//! epilogue arithmetic raises the flags of its live lanes only. A call leaves
//! FPSR = the caller's flags | the flags of the epilogues of its depth
//! blocks (M0 none, M1 those of `acc + C`, M2 of `acc · alpha`, M3 of
//! `acc · alpha` and `t + C · beta`).
//!
//! # Generated text
//!
//! The lines between each kernel's `@generated` markers below are the output
//! of `crates/oxiblas-blas/tools/gen_sme2_few_rows.py`. Edit the generator,
//! not the lines; this command, run from the workspace root, proves that they
//! are current (exit status 0, or a diff and status 1):
//!
//! ```text
//! python3 -I crates/oxiblas-blas/tools/gen_sme2_few_rows.py --check
//! ```
//!
//! # Entry points
//!
//! - [`few_rows_gemm`]: `C = alpha · A · op(B)[.., cols] + beta · C` for
//!   `1 ≤ m ≤` [`max_rows`], whole depth, KC step `kc`; `false` with C
//!   untouched when the arguments do not fit.
//! - [`self_check`]: the first-use check for the detection, the kernels
//!   against the scalar model [`oracle_few_rows`] on a few fixed blocks.
//!
//! Neither is wired into the driver yet (that is the driver-policy step);
//! the tests are in `tests_few_rows.rs`.

#[cfg(test)]
use std::cell::Cell;

use core::mem::{offset_of, size_of};

use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

use super::args::{Epilogue, MR, MacroArgs, PREFETCH_OFF, SmeElement};
use super::asm::sme_kernels;
use super::oracle::{oracle_block, same_bits_or_nan};
use super::pack::{ColMajor, Span};
use crate::level3::trsm::Trans;

/// The argument block of one few-rows kernel call: the whole product
/// `C[0..m, 0..n] = alpha · A · op(B) + beta · C` with the caller's KC step.
///
/// `#[repr(C)]`, every field 8 bytes, the same offsets for `f32` and `f64`.
/// All strides are in ELEMENTS.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct FewRowsArgs<T> {
    /// `A(0, 0)`: `A(i, p) = a[i + p · lda]` for `i < m`, `p < k`.
    pub(super) a: *const T,
    /// Elements between depth steps `p` and `p + 1` of A.
    pub(super) lda: usize,
    /// `op(B)(0, 0)` of the column span: `B(p, j) = b[p + j · ldb]` when
    /// `b_trans == 0`, `b[j + p · ldb]` when `b_trans == 1`.
    pub(super) b: *const T,
    /// Leading dimension of the stored B.
    pub(super) ldb: usize,
    /// 0: op(B) = B (column-major); 1: op(B) = B^T (rows of op(B) contiguous).
    pub(super) b_trans: u64,
    /// `C(0, 0)`: `C(i, j) = c[i + j · ldc]`.
    pub(super) c: *mut T,
    /// Elements between consecutive columns of C (`≥ m`).
    pub(super) ldc: usize,
    /// Rows (`1 ≤ m ≤ max_rows`).
    pub(super) m: usize,
    /// Columns (`≥ 1`).
    pub(super) n: usize,
    /// Depth (`≥ 1`).
    pub(super) k: usize,
    /// The caller's KC step (`1 ≤ kc ≤ k`).
    pub(super) kc: usize,
    /// Bit pattern of alpha (`f32`: zero-extended).
    pub(super) alpha_bits: u64,
    /// Bit pattern of beta, the effective beta of the first depth block.
    pub(super) beta_bits: u64,
    /// Bit pattern of one, the effective beta of every later depth block.
    pub(super) one_bits: u64,
    /// [`Epilogue`] of the first depth block: `select(alpha, beta)`.
    pub(super) mode_first: u64,
    /// [`Epilogue`] of every later depth block: `select(alpha, 1)`.
    pub(super) mode_later: u64,
}

/// Byte offset of `a`.
pub(super) const OFF_A: usize = offset_of!(FewRowsArgs<f32>, a);
/// Byte offset of `lda`.
pub(super) const OFF_LDA: usize = offset_of!(FewRowsArgs<f32>, lda);
/// Byte offset of `b`.
pub(super) const OFF_B: usize = offset_of!(FewRowsArgs<f32>, b);
/// Byte offset of `ldb`.
pub(super) const OFF_LDB: usize = offset_of!(FewRowsArgs<f32>, ldb);
/// Byte offset of `b_trans`.
pub(super) const OFF_B_TRANS: usize = offset_of!(FewRowsArgs<f32>, b_trans);
/// Byte offset of `c`.
pub(super) const OFF_C: usize = offset_of!(FewRowsArgs<f32>, c);
/// Byte offset of `ldc`.
pub(super) const OFF_LDC: usize = offset_of!(FewRowsArgs<f32>, ldc);
/// Byte offset of `m`.
pub(super) const OFF_M: usize = offset_of!(FewRowsArgs<f32>, m);
/// Byte offset of `n`.
pub(super) const OFF_N: usize = offset_of!(FewRowsArgs<f32>, n);
/// Byte offset of `k`.
pub(super) const OFF_K: usize = offset_of!(FewRowsArgs<f32>, k);
/// Byte offset of `kc`.
pub(super) const OFF_KC: usize = offset_of!(FewRowsArgs<f32>, kc);
/// Byte offset of `alpha_bits`.
pub(super) const OFF_ALPHA: usize = offset_of!(FewRowsArgs<f32>, alpha_bits);
/// Byte offset of `beta_bits`.
pub(super) const OFF_BETA: usize = offset_of!(FewRowsArgs<f32>, beta_bits);
/// Byte offset of `one_bits`.
pub(super) const OFF_ONE: usize = offset_of!(FewRowsArgs<f32>, one_bits);
/// Byte offset of `mode_first`.
pub(super) const OFF_MODE_FIRST: usize = offset_of!(FewRowsArgs<f32>, mode_first);
/// Byte offset of `mode_later`.
pub(super) const OFF_MODE_LATER: usize = offset_of!(FewRowsArgs<f32>, mode_later);

impl<T: SmeElement> FewRowsArgs<T> {
    /// A block with the epilogue modes chosen from `alpha` and `beta`
    /// (`select(alpha, beta)` for the first depth block, `select(alpha, 1)`
    /// for the later ones) and the bit patterns of alpha, beta and one.
    pub(super) fn new(
        (a, lda): (*const T, usize),
        (b, ldb, b_trans): (*const T, usize, bool),
        (c, ldc): (*mut T, usize),
        (m, n, k, kc): (usize, usize, usize, usize),
        alpha: T,
        beta: T,
    ) -> Self {
        Self {
            a,
            lda,
            b,
            ldb,
            b_trans: u64::from(b_trans),
            c,
            ldc,
            m,
            n,
            k,
            kc,
            alpha_bits: alpha.bits(),
            beta_bits: beta.bits(),
            one_bits: T::ONE.bits(),
            mode_first: Epilogue::select(alpha, beta) as u64,
            mode_later: Epilogue::select(alpha, T::ONE) as u64,
        }
    }
}

/// The element types of the few-rows kernels, with their kernel.
pub(super) trait SmeFewRows: SmeElement + Field {
    /// Most rows the kernel serves: 48 (`f32`, three `.s` tiles) or 32
    /// (`f64`, four `.d` tiles).
    const MAX_ROWS: usize;
    /// The guard pattern of the first-use check: a signalling NaN.
    const CHECK_GUARD: Self;
    /// What C holds in the first-use check when the first block does not
    /// read it: another signalling NaN, which must not leak into C.
    const CHECK_UNREAD: Self;

    /// `x` rounded to the type.
    fn of_f64(x: f64) -> Self;

    /// Runs the few-rows kernel of the type.
    ///
    /// # Safety
    ///
    /// The `# Safety` section of `sme2_few_rows_f32` / `sme2_few_rows_f64`.
    unsafe fn kernel(args: &FewRowsArgs<Self>);
}

impl SmeFewRows for f32 {
    const MAX_ROWS: usize = 48;
    const CHECK_GUARD: Self = f32::from_bits(0x7fa0_beef);
    const CHECK_UNREAD: Self = f32::from_bits(0x7f80_0bad);

    fn of_f64(x: f64) -> Self {
        x as f32
    }

    unsafe fn kernel(args: &FewRowsArgs<f32>) {
        // SAFETY: the caller upholds this function's contract, which is the
        // kernel's.
        unsafe { sme2_few_rows_f32(args) }
    }
}

impl SmeFewRows for f64 {
    const MAX_ROWS: usize = 32;
    const CHECK_GUARD: Self = f64::from_bits(0x7ff4_0000_0000_beef);
    const CHECK_UNREAD: Self = f64::from_bits(0x7ff0_0000_0000_0bad);

    fn of_f64(x: f64) -> Self {
        x
    }

    unsafe fn kernel(args: &FewRowsArgs<f64>) {
        // SAFETY: the caller upholds this function's contract, which is the
        // kernel's.
        unsafe { sme2_few_rows_f64(args) }
    }
}

/// Most rows the few-rows kernel of `T` serves: 48 (`f32`), 32 (`f64`).
pub(super) const fn max_rows<T: SmeFewRows>() -> usize {
    T::MAX_ROWS
}

#[cfg(test)]
thread_local! {
    /// Few-rows kernel calls made on this thread (test builds only). One
    /// kernel call is one `asm!` block, which is one streaming region (the
    /// frame of `asm.rs` has exactly one `smstart` and one `smstop`), so this
    /// counts the streaming regions of [`few_rows_gemm`] exactly, unlike the
    /// process-wide `test_support::kernel_entries`.
    static REGIONS: Cell<u64> = const { Cell::new(0) };
}

/// The few-rows kernel calls made on this thread so far (test builds only).
#[cfg(test)]
pub(super) fn regions_on_this_thread() -> u64 {
    REGIONS.with(Cell::get)
}

/// Whether `span` is non-empty and lies inside `0..total`.
fn inside(span: Span, total: usize) -> bool {
    span.len > 0
        && span
            .start
            .checked_add(span.len)
            .is_some_and(|end| end <= total)
}

/// The argument block of [`few_rows_gemm`] for these operands, or `None` when
/// they do not fit (see [`few_rows_gemm`]). Computing it reads and writes
/// nothing: it only checks the views and takes their addresses.
///
/// Strides that address nothing are normalised (the depth stride of A when
/// `k == 1`, the stride of B between the elements of a dimension of length 1,
/// the column stride of C when `n == 1`), so that a view whose unused stride
/// is arbitrary is accepted; every other stride must fit the kernel's byte
/// arithmetic (`NR` columns, `kc` depth steps), or the call declines.
pub(super) fn few_rows_args<T: SmeFewRows>(
    a: &MatRef<'_, T>,
    b: &ColMajor<'_, T>,
    cols: Span,
    alpha: T,
    beta: T,
    kc: usize,
    c: &mut MatMut<'_, T>,
) -> Option<FewRowsArgs<T>> {
    let (m, k, n) = (a.nrows(), a.ncols(), cols.len);
    if m == 0 || m > T::MAX_ROWS || k == 0 || n == 0 || kc == 0 || alpha == T::ZERO {
        return None;
    }
    // A column-major (a single row has no row step).
    if m > 1 && a.col_stride() != 1 {
        return None;
    }
    if b.rows() != k || !inside(cols, b.cols()) {
        return None;
    }
    if c.nrows() != m || c.ncols() != n {
        return None;
    }
    let kc = kc.min(k);
    let lda = if k == 1 { 0 } else { a.row_stride() };
    let stored = b.stored();
    let b_trans = matches!(b.trans(), Trans::Trans);
    // NoTrans: op(B)(p, j) = stored(p, cols.start + j); the stride is used
    // between columns. Trans: op(B)(p, j) = stored(cols.start + j, p); the
    // stride is used between depth steps.
    let (b_origin, ldb) = if b_trans {
        (
            stored.ptr_at(cols.start, 0),
            if k == 1 { 0 } else { stored.row_stride() },
        )
    } else {
        (
            stored.ptr_at(0, cols.start),
            if n == 1 { 0 } else { stored.row_stride() },
        )
    };
    let ldc = if n == 1 {
        m
    } else if c.row_stride() >= m {
        c.row_stride()
    } else {
        return None;
    };
    // The kernel's byte arithmetic: A advances by `kc · lda`, B by `NR · ldb`
    // (a panel) or `ldb` (a column or a depth step), C by `NR · ldc`.
    let size = size_of::<T>();
    let fits = |stride: usize, steps: usize| {
        stride
            .checked_mul(size)
            .and_then(|bytes| bytes.checked_mul(steps))
            .is_some()
    };
    if !fits(lda, kc) || !fits(ldb, T::NR) || !fits(ldc, T::NR) {
        return None;
    }
    Some(FewRowsArgs::new(
        (a.as_ptr(), lda),
        (b_origin, ldb, b_trans),
        (c.as_mut_ptr(), ldc),
        (m, n, k, kc),
        alpha,
        beta,
    ))
}

/// `C = alpha · A · op(B)[.., cols] + beta · C` for the columns `cols` of
/// op(B) and the matching `m × cols.len` view `c` (column-major,
/// `row_stride ≥ m`), whole depth, KC step `kc`, in ONE streaming region.
///
/// `a` is op(A) as a column-major `m × k` view (`col_stride == 1`,
/// `1 ≤ m ≤` [`max_rows`]); `b` is op(B) (`k × ncols`), read in place in
/// either orientation. The depth partition is `(0..k).step_by(kc)` with the
/// effective beta `beta` for the first block and `1` afterwards, exactly as
/// the blocked driver, so C is bit for bit the blocked result (NaN payloads
/// aside).
///
/// Returns `false`, with C untouched and no streaming code executed, when the
/// arguments do not fit: `m == 0`, `m > max_rows`, `k == 0`, an empty or
/// out-of-range `cols`, `kc == 0`, `alpha == 0`, `b.rows() != k`, a C view
/// of another shape, a C whose columns overlap (`row_stride < m` with more
/// than one column), A with a non-unit row step, or strides too large for
/// the kernel's arithmetic. Otherwise `true`: C holds the result.
///
/// # Safety
///
/// - `detect::thread_ready()` returned `true` on this thread and
///   `detect::enabled::<T>()` holds (for `f64` this includes
///   FEAT_SME_F64F64): the kernels' contract.
/// - C overlaps neither A nor B. (The views guarantee every address the
///   kernel forms: it reads no A row `≥ m`, nothing of B outside
///   `op(B)[0..k, cols]`, and reads and writes C only inside the view.)
pub(super) unsafe fn few_rows_gemm<T: SmeFewRows>(
    a: MatRef<'_, T>,
    b: &ColMajor<'_, T>,
    cols: Span,
    alpha: T,
    beta: T,
    kc: usize,
    mut c: MatMut<'_, T>,
) -> bool {
    let Some(args) = few_rows_args(&a, b, cols, alpha, beta, kc, &mut c) else {
        return false;
    };
    #[cfg(test)]
    REGIONS.with(|regions| regions.set(regions.get() + 1));
    // SAFETY: the kernel's contract holds:
    // - the thread is ready and `T` is enabled (this function's contract);
    // - `few_rows_args` checked `1 ≤ m ≤ max_rows`, `n, k, kc ≥ 1`
    //   (`kc ≤ k`), `ldc ≥ m` (or a single column), and took the modes from
    //   `Epilogue::select`;
    // - A, op(B)[.., cols] and C are the views' elements: `a` is column-major
    //   `m × k` with depth stride `lda`, `b_origin` is `op(B)(0, cols.start)`
    //   with the stored matrix's leading dimension, `c` is `m × n` with
    //   `ldc`; the strides normalised to 0 belong to dimensions of length 1,
    //   where the kernel never steps by them before a dereference;
    // - C is borrowed exclusively and overlaps neither A nor B (contract).
    unsafe { T::kernel(&args) };
    true
}

/// A kernel of the few-rows signature: a real kernel, the scalar model
/// [`oracle_few_rows`], or a test's stand-in.
pub(super) type FewRowsKernel<T> = unsafe fn(&FewRowsArgs<T>);

/// Writes the value of the module docs into every element of `C[0..m, 0..n]`
/// of `args`, in plain Rust: for each depth block, op(B)'s block is copied
/// into the plain packed layout of `args.rs` and `oracle::oracle_block` runs
/// with A in place (`a_depth = lda`, `a_panel = 64`, valid for `m ≤ 64`) and
/// the block's mode and effective beta. It has the kernels' signature, so it
/// can stand in for one.
///
/// # Safety
///
/// Every address the layout of [`FewRowsArgs`] names for `i < m`, `p < k`,
/// `j < n` must be valid (C for reads and writes, not overlapping A or B);
/// `1 ≤ m ≤ 64`, `kc ≥ 1`.
pub(super) unsafe fn oracle_few_rows<T: SmeElement>(args: &FewRowsArgs<T>) {
    let nr = T::NR;
    let (m, n, k) = (args.m, args.n, args.k);
    let kc = args.kc.max(1);
    let mut packed = Vec::new();
    let mut p0 = 0;
    while p0 < k {
        let d = kc.min(k - p0);
        packed.clear();
        packed.resize(n.div_ceil(nr) * nr * d, T::ZERO);
        for j in 0..n {
            for p in 0..d {
                let at = if args.b_trans == 0 {
                    (p0 + p) + j * args.ldb
                } else {
                    j + (p0 + p) * args.ldb
                };
                // SAFETY: `p0 + p < k` and `j < n`: the address of
                // `op(B)(p0 + p, j)`, valid for reads by the caller's contract.
                let value = unsafe { *args.b.add(at) };
                if let Some(slot) = packed.get_mut((j / nr) * d * nr + p * nr + j % nr) {
                    *slot = value;
                }
            }
        }
        let (beta_bits, mode) = if p0 == 0 {
            (args.beta_bits, args.mode_first)
        } else {
            (args.one_bits, args.mode_later)
        };
        let block = MacroArgs {
            // SAFETY: `p0 < k`, so `A(0, p0)` is inside A (caller's contract).
            a: unsafe { args.a.add(p0 * args.lda) },
            a_depth: args.lda,
            a_panel: MR,
            b: packed.as_ptr(),
            b_panel: d * nr,
            c: args.c,
            ldc: args.ldc,
            m,
            n,
            kc: d,
            alpha_bits: args.alpha_bits,
            beta_bits,
            mode,
            a_pack: core::ptr::null_mut(),
            prefetch: PREFETCH_OFF,
        };
        // SAFETY: A in place: `A(i, p0 + p)` for `i < m ≤ 64`, `p < d` is
        // `a[i + (p0 + p) · lda]`, valid by the caller's contract; the packed
        // copy holds whole zero-padded panels of depth `d`; C as the caller
        // guarantees, overlapping neither A nor the copy.
        unsafe { oracle_block(&block) };
        p0 += d;
    }
}

/// One block of the first-use check, with its row count given relative to
/// the type (so the same list covers every depth-loop variant of both
/// kernels).
#[derive(Debug, Clone, Copy)]
struct CheckCase {
    /// Rows: `tiles` tiles of `TILE_ROWS` rows, minus `short` (`m ≥ 1`).
    tiles: usize,
    /// Rows below a whole number of tiles.
    short: usize,
    /// Columns: `panels` panels of `NR`, plus `extra`.
    panels: usize,
    /// Columns beyond whole panels.
    extra: usize,
    /// Depth in units of the transposer's chunk (`TILE_ROWS` steps), plus
    /// `depth_extra`.
    chunks: usize,
    /// Depth steps beyond whole chunks.
    depth_extra: usize,
    /// KC step: `chunks` of it plus `kc_extra` (0 + 0 = the whole depth).
    kc_chunks: usize,
    /// KC steps beyond whole chunks.
    kc_extra: usize,
    /// op(B) = B^T.
    trans: bool,
    /// Index into [`CHECK_SCALARS`].
    scalars: usize,
}

/// `(alpha, beta)` of the check: M3 / M3, M0 / M1, M2, M1, M3 with
/// `beta == 1`, M2 / M3.
const CHECK_SCALARS: [(f64, f64); 6] = [
    (0.5, -0.5),
    (1.0, 0.0),
    (-2.0, 0.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (0.75, -0.0),
];

/// A [`CheckCase`]: `[tiles, short]`, `[panels, extra]`,
/// `[chunks, depth_extra]`, `[kc_chunks, kc_extra]`, `trans`, `scalars`.
const fn check_case(
    [tiles, short]: [usize; 2],
    [panels, extra]: [usize; 2],
    [chunks, depth_extra]: [usize; 2],
    [kc_chunks, kc_extra]: [usize; 2],
    trans: bool,
    scalars: usize,
) -> CheckCase {
    CheckCase {
        tiles,
        short,
        panels,
        extra,
        chunks,
        depth_extra,
        kc_chunks,
        kc_extra,
        trans,
        scalars,
    }
}

/// The blocks of [`self_check`]: every depth-loop variant (1 to 3 or 4 live
/// tiles) in both orientations, whole and partial tiles, panels and chunks,
/// several depth blocks with a short last one, and every mode of the first
/// and of the later blocks.
///
/// Every variant is also checked at both of its row bounds: its highest row
/// count (`L` whole tiles) and its lowest, `(L - 1) · TILE_ROWS + 1` rows. A
/// kernel whose variant choice is off by one row at a boundary (`m` rows
/// just above it run by the variant below, so the last row is never
/// computed) fails only at that lowest count. `short` counts rows, so the
/// lowest counts come from per-type entries: `[2, 15]` and `[3, 15]` give
/// `f32` 17 and 33 rows, `[2, 7]`, `[3, 15]`, `[3, 7]` and `[4, 7]` give
/// `f64` 9, 9, 17 and 25 rows (the same entries give the other type other
/// row counts).
const CHECK_CASES: [CheckCase; 16] = [
    check_case([1, 15], [0, 3], [0, 5], [0, 0], false, 2),
    check_case([1, 0], [1, 1], [2, 3], [1, 1], true, 0),
    check_case([2, 1], [2, 3], [2, 3], [1, 1], false, 0),
    check_case([2, 7], [1, 0], [1, 0], [0, 0], true, 3),
    check_case([3, 0], [0, 5], [1, 7], [0, 7], false, 1),
    check_case([3, 2], [1, 2], [1, 1], [1, 0], true, 4),
    check_case([4, 0], [1, 1], [3, 1], [2, 0], false, 5),
    check_case([4, 3], [0, 7], [1, 2], [0, 0], true, 2),
    check_case([1, 0], [0, 1], [0, 1], [0, 0], false, 1),
    check_case([2, 0], [2, 0], [1, 1], [0, 3], true, 5),
    check_case([3, 9], [1, 1], [0, 3], [0, 1], false, 3),
    check_case([4, 5], [1, 0], [2, 0], [1, 0], true, 1),
    check_case([2, 15], [1, 3], [1, 2], [0, 0], true, 3),
    check_case([3, 15], [2, 1], [2, 1], [1, 3], false, 4),
    check_case([3, 7], [0, 6], [2, 5], [1, 2], true, 5),
    check_case([4, 7], [1, 2], [1, 7], [0, 5], false, 0),
];

/// Guard elements before and after C and after the stored operands. Every
/// element of the check's buffers outside the block of C (the guards, the
/// padding rows of A, B and C, and A and B themselves) must keep its bits.
const CHECK_GUARD: usize = 3;

/// A deterministic, inexact value in `[-1, 1)` for element `index` of a
/// buffer.
fn check_value<T: SmeFewRows>(index: usize, salt: u64) -> T {
    let mixed = (index as u64)
        .wrapping_add(salt.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let fraction = (mixed >> 40) as f64 / (1u64 << 24) as f64;
    T::of_f64(2.0 * fraction - 1.0 + 1.0 / 3.0e7)
}

/// Runs one block of the check with `kernel`; `true` when C equals the
/// scalar model (bitwise, NaN payloads aside), every element of the C buffer
/// outside the block (the guards before and after it, its padding rows) kept
/// its bits, and the A and B buffers (operands, padding and guards) are
/// bitwise unchanged.
///
/// # Safety
///
/// `kernel` must be callable with any valid few-rows argument block on this
/// thread (for the real kernels: their contract).
unsafe fn check_block<T: SmeFewRows>(kernel: FewRowsKernel<T>, case: &CheckCase) -> bool {
    let tile = T::TILE_ROWS;
    let tiles = case.tiles.min(T::MAX_ROWS / tile);
    let m = (tiles * tile).saturating_sub(case.short).max(1);
    let n = case.panels * T::NR + case.extra;
    let k = case.chunks * tile + case.depth_extra;
    let kc = match case.kc_chunks * tile + case.kc_extra {
        0 => k,
        kc => kc.min(k),
    };
    let (alpha, beta) = CHECK_SCALARS[case.scalars % CHECK_SCALARS.len()];
    let (alpha, beta) = (T::of_f64(alpha), T::of_f64(beta));
    if n == 0 || k == 0 {
        return false;
    }
    let guard = T::CHECK_GUARD;
    // A: m x k at offset 1 of a column-major buffer with lda = m + 2.
    let lda = m + 2;
    let mut a = vec![guard; 1 + lda * k + CHECK_GUARD];
    // B: stored k x (n + 1) (NoTrans, ldb = k + 1) or (n + 1) x k (Trans,
    // ldb = n + 2); op(B)'s columns start at stored column / row 1.
    let (stored_rows, stored_cols) = if case.trans {
        (n + 2, k)
    } else {
        (k + 1, n + 1)
    };
    let ldb = stored_rows;
    let mut b = vec![guard; ldb * stored_cols + CHECK_GUARD];
    for i in 0..m {
        for p in 0..k {
            if let Some(slot) = a.get_mut(1 + i + p * lda) {
                *slot = check_value::<T>(i * k + p, 1);
            }
        }
    }
    for p in 0..k {
        for j in 0..n {
            let at = if case.trans {
                (j + 1) + p * ldb
            } else {
                p + (j + 1) * ldb
            };
            if let Some(slot) = b.get_mut(at) {
                *slot = check_value::<T>(p * n + j, 2);
            }
        }
    }
    let b_origin = if case.trans { 1 } else { ldb };
    // C: m x n at offset CHECK_GUARD with ldc = m + 1: guards before it, the
    // rows m..ldc and a tail of guards. C starts as a NaN when the first
    // block does not read it.
    let ldc = m + 1;
    let c_off = CHECK_GUARD;
    let reads_c = Epilogue::select(alpha, beta).reads_c();
    let mut want = vec![guard; c_off + ldc * n + CHECK_GUARD];
    for j in 0..n {
        for i in 0..m {
            if let Some(slot) = want.get_mut(c_off + i + j * ldc) {
                *slot = if reads_c {
                    check_value::<T>(i + j * m, 3)
                } else {
                    T::CHECK_UNREAD
                };
            }
        }
    }
    let mut got = want.clone();
    let a_before: Vec<u64> = a.iter().map(|value| value.bits()).collect();
    let b_before: Vec<u64> = b.iter().map(|value| value.bits()).collect();
    // Mutable provenance: a faulty kernel (or a test's stand-in) that writes
    // A or B must hit memory the check may write, so that the comparison
    // below can see it.
    let a_ptr = a.as_mut_ptr().wrapping_add(1).cast_const();
    let b_ptr = b.as_mut_ptr().wrapping_add(b_origin).cast_const();
    let args_for = |c: &mut Vec<T>| {
        FewRowsArgs::new(
            (a_ptr, lda),
            (b_ptr, ldb, case.trans),
            (c.as_mut_ptr().wrapping_add(c_off), ldc),
            (m, n, k, kc),
            alpha,
            beta,
        )
    };
    let want_args = args_for(&mut want);
    // SAFETY: the buffers hold A (`1 + i + p · lda`), op(B) and C
    // (`c_off + i + j · ldc`) for every index of the block, C overlaps
    // neither; `m ≤ max_rows ≤ 64`.
    unsafe { oracle_few_rows(&want_args) };
    let got_args = args_for(&mut got);
    // SAFETY: as above, and the caller vouches for `kernel` on this thread.
    unsafe { kernel(&got_args) };
    let block_equal = (0..n).all(|j| {
        let column = c_off + j * ldc..c_off + j * ldc + m;
        match (got.get(column.clone()), want.get(column)) {
            (Some(got), Some(want)) => same_bits_or_nan(got, want),
            _ => false,
        }
    });
    let guard_bits = guard.bits();
    let guards_kept = got
        .iter()
        .enumerate()
        .filter(|(at, _)| *at < c_off || *at >= c_off + ldc * n || (at - c_off) % ldc >= m)
        .all(|(_, value)| value.bits() == guard_bits);
    let unchanged = |buffer: &[T], before: &[u64]| {
        buffer.len() == before.len()
            && buffer
                .iter()
                .zip(before)
                .all(|(value, bits)| value.bits() == *bits)
    };
    block_equal && guards_kept && unchanged(&a, &a_before) && unchanged(&b, &b_before)
}

/// The check of [`self_check`] for one type and one kernel: every block of
/// [`CHECK_CASES`] equal to the scalar model. Tests pass deliberately wrong
/// stand-ins to prove that it can fail.
///
/// # Safety
///
/// `kernel` must be callable with any valid few-rows argument block on this
/// thread (for the real kernels: their contract).
pub(super) unsafe fn self_check_with<T: SmeFewRows>(kernel: FewRowsKernel<T>) -> bool {
    // SAFETY: forwarded from this function's contract.
    CHECK_CASES
        .iter()
        .all(|case| unsafe { check_block(kernel, case) })
}

/// First-use check of the few-rows GEMM, as `pack_za::self_check`: runs
/// the `f32` kernel, and the `f64` kernel when `f64_present`, on the fixed
/// blocks of [`CHECK_CASES`] and compares C with the scalar model
/// [`oracle_few_rows`] (and every other element of the C, A and B buffers
/// with its bits before the call: guards, padding, the operands); `true` when
/// all are equal.
///
/// The detection runs it (`detect::run_checks`, third of the four checks)
/// on the first detection call (`detect::hw` or `detect::hw_now`) from a
/// ready thread while the verdict is open. The check leaves
/// cumulative floating-point flags (the model's arithmetic and the kernels'
/// epilogues); the detection runs it with FPCR = 0 and restores the
/// caller's FPCR and FPSR afterwards. It calls nothing that runs or reads
/// the detection, creates no thread and never panics.
///
/// Cost: one kernel call (one streaming region) per block of
/// [`CHECK_CASES`], 16 for `f32` and 16 more for `f64`, on blocks with
/// `m ≤ 48`, `n ≤ 35`, `k ≤ 49` (`f32`) and `m ≤ 32`, `n ≤ 19`, `k ≤ 25`
/// (`f64`); each block allocates its A, B and two C buffers of a few
/// kilobytes, and the model a packed copy of B per depth block.
///
/// # Safety
///
/// The kernels' contract: SME2 with a 512-bit streaming vector length is
/// present (for `f64_present`, FEAT_SME_F64F64 too), and the calling thread
/// was just found in the state that `detect::thread_ready()` checks
/// (outside the detection: `detect::thread_ready()` returned `true`; the
/// detection reads the state with `detect::probe_thread_state()`).
pub(super) unsafe fn self_check(f64_present: bool) -> bool {
    // SAFETY: the caller upholds the kernels' contract on this thread, and
    // every block of the check is a valid argument block (`check_block`).
    unsafe {
        self_check_with::<f32>(sme2_few_rows_f32)
            && (!f64_present || self_check_with::<f64>(sme2_few_rows_f64))
    }
}

sme_kernels! {
    /// The `f32` few-rows kernel: the whole product `C[0..m, 0..n] = alpha ·
    /// A · op(B) + beta · C` of `args` (`1 ≤ m ≤ 48`) in one streaming
    /// region, to the value of the module docs (per depth block exactly the
    /// macro-kernel's arithmetic).
    ///
    /// # Loop nest
    ///
    /// ```text
    /// p1, p2, p3 = rows < m of tiles 0, 1, 2; pn9 = rows < m of tiles 0..3
    /// for p0 in (0..k).step_by(kc)                     label 70, d = min(kc, k - p0)
    ///   for j0 in (0..n).step_by(16)                   label 20, ncols = min(16, n - j0)
    ///     p6 = columns j0.. below n
    ///     L = ceil(m / 16)                              variant 4(2L-1): 41, 43, 45
    ///     zero za0..za(L-1)
    ///     NoTrans (b_trans == 0):
    ///       chunks of 16 depth steps                    label 2L2
    ///         p5 = depth steps left (whilelt 0, left)
    ///         for c in 0..ncols: B(p.., j0 + c) -> za3h.s[c]  label 2L3 (p5/z)
    ///         whole chunk: 4 x (mova 4 vertical slices,  label 2L4
    ///           4 depth steps of A -> FMOPA)
    ///         partial chunk: 1 vertical slice per step   label 2L5
    ///     Trans (b_trans == 1):
    ///       pairs of depth steps: B rows (p6/z), A       label 2L8
    ///       one more step when d is odd                  label 2L7
    ///     epilogue of the block's mode                   labels 50 (dispatch), 51..54
    ///       for c in 0..ncols, tile t while 16t < m: slice c of za<t>, predicate p<t+1>
    ///   next column panel                                label 59 (66, 67: B panel step)
    /// next depth block: mode = mode_later, beta = one
    /// ```
    ///
    /// # Register map
    ///
    /// | Register | Use |
    /// |---|---|
    /// | `x1`, `x2` | `a`, `lda` in bytes |
    /// | `x3`, `x4`, `x5` | `b`, `ldb` in bytes, `b_trans` |
    /// | `x6`, `x7` | `c`, `ldc` in bytes |
    /// | `x8` | `ncols = min(16, n - j0)` |
    /// | `x9` | `C(0, j0)` |
    /// | `x10` | `op(B)(p0, j0)` |
    /// | `x11` | depth steps of the block not yet run |
    /// | `w12` | slice index: column of the panel (B loads, epilogue); `x12` also the 0 of `whilelt p5` |
    /// | `w13` | vertical slice index: depth step of the chunk; `x13` the C address in the epilogue |
    /// | `x14` | A at the current depth step; 16 (tile 1 row offset) in the epilogue |
    /// | `x15` | B of the chunk (NoTrans) or of the depth step (Trans); 32 in the epilogue |
    /// | `x16` | scratch: column address of a B load, double steps left, bit patterns |
    /// | `x20`, `x21`, `x22`, `x23` | `m`, `n`, `k`, `kc` |
    /// | `x24` | `p0` |
    /// | `x25` | mode of the current depth block (0 M0, 1 M1, 2 M2, 3 M3) |
    /// | `x26` | `d` |
    /// | `x27` | `A(0, p0)` |
    /// | `x28` | `j0` |
    /// | `z0`–`z3`, `z4`–`z7` | A vectors of two depth steps (tile `t` in `z{t}` / `z{4+t}`) |
    /// | `z8`–`z11` | B rows of up to four depth steps |
    /// | `z16`, `z17` | epilogue: the moved slice, the loaded C |
    /// | `z28`, `z29` | alpha, the block's beta (beta, then one) |
    /// | `p0` | all-true |
    /// | `p1`–`p3` | rows below `m` of tiles 0–2 |
    /// | `p5` | live depth steps of the chunk (NoTrans B column loads) |
    /// | `p6` | live columns of the panel (Trans B row loads) |
    /// | `pn9` | rows below `m` of tiles 0–3 (multi-vector A loads) |
    /// | `za0`–`za2` | the live C tiles; `za3` the transposer of NoTrans B |
    ///
    /// Epilogue of tile `t`, column `c` (`pt` = `p<t+1>`, `addr` =
    /// `C(16t, j0 + c)`): M0 `st1w {za<t>h.s[w12, 0]}, pt, [addr]`; M1 MOVA,
    /// `ld1w` of C (`pt/z`), `fadd z16.s, pt/m, z16.s, z17.s`, store; M2
    /// MOVA, `fmul` by `z28`, store; M3 MOVA, `fmul` by `z28`, `ld1w` of C,
    /// `fmla z16.s, pt/m, z17.s, z29.s`, store. C is read only in M1 and M3.
    ///
    /// It reads no A row `≥ m` and nothing of B outside `op(B)[0..k, 0..n]`,
    /// reads C only when the block's mode reads it, writes only
    /// `C[0..m, 0..n]`, never writes the argument block, and leaves through
    /// the frame's label 90.
    ///
    /// # Safety
    ///
    /// - `detect::thread_ready()` must have returned `true` on this thread
    ///   (SME2 with a 512-bit streaming vector length, not in streaming
    ///   mode, no live ZA state, no pending lazy save).
    /// - `1 ≤ m ≤ 48`, `n ≥ 1`, `1 ≤ kc ≤ k`, `ldc ≥ m` (when `n > 1`),
    ///   `b_trans` 0 or 1, and the modes must be `Epilogue::select(alpha,
    ///   beta)` and `Epilogue::select(alpha, 1)`, `one_bits` the bits of 1.
    /// - Every address the layout of [`FewRowsArgs`] names for `i < m`,
    ///   `p < k`, `j < n` must be valid (C for reads and writes); C must not
    ///   overlap A, B or the argument block.
    pub(super) streaming fn sme2_few_rows_f32(args: &FewRowsArgs<f32>) {
        extensions: [],
        // @generated begin: tools/gen_sme2_few_rows.py, sme2_few_rows_f32 (edit the generator, not these lines)
        consts: [
            off_a = OFF_A,
            off_lda = OFF_LDA,
            off_b = OFF_B,
            off_ldb = OFF_LDB,
            off_b_trans = OFF_B_TRANS,
            off_c = OFF_C,
            off_ldc = OFF_LDC,
            off_m = OFF_M,
            off_n = OFF_N,
            off_k = OFF_K,
            off_kc = OFF_KC,
            off_alpha = OFF_ALPHA,
            off_beta = OFF_BETA,
            off_one = OFF_ONE,
            off_mode_first = OFF_MODE_FIRST,
            off_mode_later = OFF_MODE_LATER,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_lda}]",
            "ldr x3, [x0, #{off_b}]",
            "ldr x4, [x0, #{off_ldb}]",
            "ldr x5, [x0, #{off_b_trans}]",
            "ldr x6, [x0, #{off_c}]",
            "ldr x7, [x0, #{off_ldc}]",
            "ldr x20, [x0, #{off_m}]",
            "ldr x21, [x0, #{off_n}]",
            "ldr x22, [x0, #{off_k}]",
            "ldr x23, [x0, #{off_kc}]",
            "ldr x25, [x0, #{off_mode_first}]",
            "lsl x2, x2, #2",
            "lsl x4, x4, #2",
            "lsl x7, x7, #2",
        ],
        body: [
            // p0 all-true; z28 alpha, z29 beta (the first depth block's).
            "ptrue p0.s",
            "ldr w16, [x0, #{off_alpha}]", "dup z28.s, w16",
            "ldr w16, [x0, #{off_beta}]", "dup z29.s, w16",
            // Rows below m: p1 (tile 0), pn9 (tiles 0-3, for the A loads), p2..p3 (tiles 1..).
            "mov x16, #0", "whilelt p1.s, x16, x20", "whilelt pn9.s, x16, x20, vlx4",
            "mov x16, #16", "whilelt p2.s, x16, x20",
            "mov x16, #32", "whilelt p3.s, x16, x20",
            // x24 = p0 (first depth step of the block), x27 = A(0, p0).
            "mov x24, #0", "mov x27, x1",
            // Depth block: x26 = d = min(kc, k - p0); x28 = j0, x9 = C(0, j0),
            // x10 = B(p0, j0) of op(B): b + p0 (NoTrans, column-major) or b + p0 * ldb (Trans).
            "70:", "sub x26, x22, x24", "cmp x26, x23", "csel x26, x26, x23, lo",
            "mov x28, #0", "mov x9, x6", "cbnz x5, 71f", "add x10, x3, x24, lsl #2", "b 20f",
            "71:", "madd x10, x24, x4, x3",
            // Column panel: x8 = ncols = min(16, n - j0), p6 = columns below n; x14 = A,
            // x15 = B at depth p0 and column j0, x11 = depth steps left in the block.
            "20:", "sub x8, x21, x28", "mov x16, #16", "cmp x8, x16", "csel x8, x8, x16, lo",
            "whilelt p6.s, x28, x21", "mov x14, x27", "mov x15, x10", "mov x11, x26",
            // Live C tiles L = ceil(m / 16) select the variant 4(2L-1).
            "cmp x20, #32", "b.hi 45f",
            "cmp x20, #16", "b.hi 43f",
            // 1 live C tile (1 to 16 rows): zero them, then the depth loop.
            "41:", "zero {{za0.s}}", "cbnz x5, 216f",
            // NoTrans B. Chunk of min(16, x11) depth steps: column c of the panel
            // (c < ncols) into horizontal slice c of za3.s, lanes of the live depth (p5).
            "212:", "mov w12, #0", "whilelt p5.s, x12, x11", "mov x16, x15",
            "213:", "ld1w {{za3h.s[w12, 0]}}, p5/z, [x16]", "add x16, x16, x4",
            "add w12, w12, #1", "cmp w12, w8", "b.lo 213b",
            // Vertical slice q of za3.s is the B row of depth step q of the chunk.
            "mov w13, #0", "cmp x11, #16", "b.lo 215f",
            // A whole chunk: four B rows per MOVA, 4 groups.
            "214:", "mova {{z8.s-z11.s}}, za3v.s[w13, 0:3]",
            "ld1w {{z0.s}}, p1/z, [x14]", "add x14, x14, x2", "ld1w {{z4.s}}, p1/z, [x14]",
            "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s",
            "ld1w {{z0.s}}, p1/z, [x14]", "add x14, x14, x2", "ld1w {{z4.s}}, p1/z, [x14]",
            "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z10.s, z0.s",
            "fmopa za0.s, p0/m, p0/m, z11.s, z4.s",
            "add w13, w13, #4", "cmp w13, #16", "b.lo 214b",
            "add x15, x15, #64", "subs x11, x11, #16", "b.ne 212b", "b 50f",
            // The last, partial chunk of the block: one depth step per B row.
            "215:", "mova z8.s, p0/m, za3v.s[w13, 0]",
            "ld1w {{z0.s}}, p1/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "add w13, w13, #1", "cmp w13, w11", "b.lo 215b", "b 50f",
            // Trans B: the B row of a depth step is contiguous; columns < n (p6). Two
            // steps per iteration, then one more when the block's depth is odd.
            "216:", "lsr x16, x11, #1", "cbz x16, 217f",
            "218:", "ld1w {{z8.s}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1w {{z0.s}}, p1/z, [x14]", "add x14, x14, x2",
            "ld1w {{z9.s}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1w {{z4.s}}, p1/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s",
            "subs x16, x16, #1", "b.ne 218b",
            "217:", "tbz x11, #0, 50f", "ld1w {{z8.s}}, p6/z, [x15]",
            "ld1w {{z0.s}}, p1/z, [x14]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s",
            "b 50f",
            // 2 live C tiles (17 to 32 rows): zero them, then the depth loop.
            "43:", "zero {{za0.s, za1.s}}", "cbnz x5, 226f",
            // NoTrans B. Chunk of min(16, x11) depth steps: column c of the panel
            // (c < ncols) into horizontal slice c of za3.s, lanes of the live depth (p5).
            "222:", "mov w12, #0", "whilelt p5.s, x12, x11", "mov x16, x15",
            "223:", "ld1w {{za3h.s[w12, 0]}}, p5/z, [x16]", "add x16, x16, x4",
            "add w12, w12, #1", "cmp w12, w8", "b.lo 223b",
            // Vertical slice q of za3.s is the B row of depth step q of the chunk.
            "mov w13, #0", "cmp x11, #16", "b.lo 225f",
            // A whole chunk: four B rows per MOVA, 4 groups.
            "224:", "mova {{z8.s-z11.s}}, za3v.s[w13, 0:3]",
            "ld1w {{z0.s-z1.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1w {{z4.s-z5.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s", "fmopa za1.s, p0/m, p0/m, z9.s, z5.s",
            "ld1w {{z0.s-z1.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1w {{z4.s-z5.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z10.s, z0.s", "fmopa za1.s, p0/m, p0/m, z10.s, z1.s",
            "fmopa za0.s, p0/m, p0/m, z11.s, z4.s", "fmopa za1.s, p0/m, p0/m, z11.s, z5.s",
            "add w13, w13, #4", "cmp w13, #16", "b.lo 224b",
            "add x15, x15, #64", "subs x11, x11, #16", "b.ne 222b", "b 50f",
            // The last, partial chunk of the block: one depth step per B row.
            "225:", "mova z8.s, p0/m, za3v.s[w13, 0]",
            "ld1w {{z0.s-z1.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "add w13, w13, #1", "cmp w13, w11", "b.lo 225b", "b 50f",
            // Trans B: the B row of a depth step is contiguous; columns < n (p6). Two
            // steps per iteration, then one more when the block's depth is odd.
            "226:", "lsr x16, x11, #1", "cbz x16, 227f",
            "228:", "ld1w {{z8.s}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1w {{z0.s-z1.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1w {{z9.s}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1w {{z4.s-z5.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s", "fmopa za1.s, p0/m, p0/m, z9.s, z5.s",
            "subs x16, x16, #1", "b.ne 228b",
            "227:", "tbz x11, #0, 50f", "ld1w {{z8.s}}, p6/z, [x15]",
            "ld1w {{z0.s-z1.s}}, pn9/z, [x14]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "b 50f",
            // 3 live C tiles (33 to 48 rows): zero them, then the depth loop.
            "45:", "zero {{za0.s, za1.s, za2.s}}", "cbnz x5, 236f",
            // NoTrans B. Chunk of min(16, x11) depth steps: column c of the panel
            // (c < ncols) into horizontal slice c of za3.s, lanes of the live depth (p5).
            "232:", "mov w12, #0", "whilelt p5.s, x12, x11", "mov x16, x15",
            "233:", "ld1w {{za3h.s[w12, 0]}}, p5/z, [x16]", "add x16, x16, x4",
            "add w12, w12, #1", "cmp w12, w8", "b.lo 233b",
            // Vertical slice q of za3.s is the B row of depth step q of the chunk.
            "mov w13, #0", "cmp x11, #16", "b.lo 235f",
            // A whole chunk: four B rows per MOVA, 4 groups.
            "234:", "mova {{z8.s-z11.s}}, za3v.s[w13, 0:3]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1w {{z4.s-z7.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z8.s, z2.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s", "fmopa za1.s, p0/m, p0/m, z9.s, z5.s",
            "fmopa za2.s, p0/m, p0/m, z9.s, z6.s",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1w {{z4.s-z7.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z10.s, z0.s", "fmopa za1.s, p0/m, p0/m, z10.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z10.s, z2.s",
            "fmopa za0.s, p0/m, p0/m, z11.s, z4.s", "fmopa za1.s, p0/m, p0/m, z11.s, z5.s",
            "fmopa za2.s, p0/m, p0/m, z11.s, z6.s",
            "add w13, w13, #4", "cmp w13, #16", "b.lo 234b",
            "add x15, x15, #64", "subs x11, x11, #16", "b.ne 232b", "b 50f",
            // The last, partial chunk of the block: one depth step per B row.
            "235:", "mova z8.s, p0/m, za3v.s[w13, 0]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z8.s, z2.s",
            "add w13, w13, #1", "cmp w13, w11", "b.lo 235b", "b 50f",
            // Trans B: the B row of a depth step is contiguous; columns < n (p6). Two
            // steps per iteration, then one more when the block's depth is odd.
            "236:", "lsr x16, x11, #1", "cbz x16, 237f",
            "238:", "ld1w {{z8.s}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1w {{z9.s}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1w {{z4.s-z7.s}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z8.s, z2.s",
            "fmopa za0.s, p0/m, p0/m, z9.s, z4.s", "fmopa za1.s, p0/m, p0/m, z9.s, z5.s",
            "fmopa za2.s, p0/m, p0/m, z9.s, z6.s",
            "subs x16, x16, #1", "b.ne 238b",
            "237:", "tbz x11, #0, 50f", "ld1w {{z8.s}}, p6/z, [x15]",
            "ld1w {{z0.s-z3.s}}, pn9/z, [x14]",
            "fmopa za0.s, p0/m, p0/m, z8.s, z0.s", "fmopa za1.s, p0/m, p0/m, z8.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z8.s, z2.s",
            "b 50f",
            // Epilogue of the 48 x 16 C tile of the column panel, by mode:
            // columns 0..ncols (w12), each tile while its first row is below m.
            "50:", "mov w12, #0", "mov x13, x9",
            "mov x14, #16", "mov x15, #32",
            "cbz x25, 51f", "cmp x25, #2", "b.lo 52f", "b.eq 53f", "b 54f",
            // M0: C = acc. Direct ZA slice stores; C is not read.
            "51:",
            "st1w {{za0h.s[w12, 0]}}, p1, [x13]",
            "cmp x20, #16", "b.ls 55f",
            "st1w {{za1h.s[w12, 0]}}, p2, [x13, x14, lsl #2]",
            "cmp x20, #32", "b.ls 55f",
            "st1w {{za2h.s[w12, 0]}}, p3, [x13, x15, lsl #2]",
            "55:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", "b.lo 51b",
            "b 59f",
            // M1: C = acc + C (SVE FADD, accumulator first).
            "52:",
            "mova z16.s, p0/m, za0h.s[w12, 0]", "ld1w {{z17.s}}, p1/z, [x13]",
            "fadd z16.s, p1/m, z16.s, z17.s", "st1w {{z16.s}}, p1, [x13]",
            "cmp x20, #16", "b.ls 56f",
            "mova z16.s, p0/m, za1h.s[w12, 0]", "ld1w {{z17.s}}, p2/z, [x13, x14, lsl #2]",
            "fadd z16.s, p2/m, z16.s, z17.s", "st1w {{z16.s}}, p2, [x13, x14, lsl #2]",
            "cmp x20, #32", "b.ls 56f",
            "mova z16.s, p0/m, za2h.s[w12, 0]", "ld1w {{z17.s}}, p3/z, [x13, x15, lsl #2]",
            "fadd z16.s, p3/m, z16.s, z17.s", "st1w {{z16.s}}, p3, [x13, x15, lsl #2]",
            "56:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", "b.lo 52b",
            "b 59f",
            // M2: C = acc * alpha (SVE FMUL); C is not read.
            "53:",
            "mova z16.s, p0/m, za0h.s[w12, 0]", "fmul z16.s, p1/m, z16.s, z28.s",
            "st1w {{z16.s}}, p1, [x13]",
            "cmp x20, #16", "b.ls 57f",
            "mova z16.s, p0/m, za1h.s[w12, 0]", "fmul z16.s, p2/m, z16.s, z28.s",
            "st1w {{z16.s}}, p2, [x13, x14, lsl #2]",
            "cmp x20, #32", "b.ls 57f",
            "mova z16.s, p0/m, za2h.s[w12, 0]", "fmul z16.s, p3/m, z16.s, z28.s",
            "st1w {{z16.s}}, p3, [x13, x15, lsl #2]",
            "57:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", "b.lo 53b",
            "b 59f",
            // M3: t = acc * alpha (SVE FMUL), C = t + C * beta (SVE FMLA, one rounding).
            "54:",
            "mova z16.s, p0/m, za0h.s[w12, 0]", "fmul z16.s, p1/m, z16.s, z28.s",
            "ld1w {{z17.s}}, p1/z, [x13]", "fmla z16.s, p1/m, z17.s, z29.s",
            "st1w {{z16.s}}, p1, [x13]",
            "cmp x20, #16", "b.ls 58f",
            "mova z16.s, p0/m, za1h.s[w12, 0]", "fmul z16.s, p2/m, z16.s, z28.s",
            "ld1w {{z17.s}}, p2/z, [x13, x14, lsl #2]", "fmla z16.s, p2/m, z17.s, z29.s",
            "st1w {{z16.s}}, p2, [x13, x14, lsl #2]",
            "cmp x20, #32", "b.ls 58f",
            "mova z16.s, p0/m, za2h.s[w12, 0]", "fmul z16.s, p3/m, z16.s, z28.s",
            "ld1w {{z17.s}}, p3/z, [x13, x15, lsl #2]", "fmla z16.s, p3/m, z17.s, z29.s",
            "st1w {{z16.s}}, p3, [x13, x15, lsl #2]",
            "58:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", "b.lo 54b",
            // Next column panel; then the next depth block, whose epilogue mode is the later
            // blocks' and whose beta is one. The end falls through to the exit label 90.
            "59:", "add x28, x28, #16", "add x9, x9, x7, lsl #4",
            "cbnz x5, 66f", "add x10, x10, x4, lsl #4", "b 67f",
            "66:", "add x10, x10, #64",
            "67:", "cmp x28, x21", "b.lo 20b",
            "add x24, x24, x23", "madd x27, x23, x2, x27",
            "ldr x25, [x0, #{off_mode_later}]",
            "ldr w16, [x0, #{off_one}]", "dup z29.s, w16",
            "cmp x24, x22", "b.lo 70b",
        ],
        // @generated end
    }

    /// The `f64` few-rows kernel: the whole product `C[0..m, 0..n] = alpha ·
    /// A · op(B) + beta · C` of `args` (`1 ≤ m ≤ 32`) in one streaming
    /// region, to the value of the module docs (per depth block exactly the
    /// macro-kernel's arithmetic).
    ///
    /// # Loop nest
    ///
    /// ```text
    /// p1..p4 = rows < m of tiles 0..3; pn9 = rows < m of tiles 0..3
    /// for p0 in (0..k).step_by(kc)                     label 70, d = min(kc, k - p0)
    ///   for j0 in (0..n).step_by(8)                    label 20, ncols = min(8, n - j0)
    ///     p6 = columns j0.. below n
    ///     L = ceil(m / 8)                               variant 4(2L-1): 41, 43, 45, 47
    ///     zero za0..za(L-1)
    ///     NoTrans (b_trans == 0):
    ///       chunks of 8 depth steps                     label 2L2
    ///         p5 = depth steps left (whilelt 0, left)
    ///         for c in 0..ncols: B(p.., j0 + c) -> za4h.d[c]  label 2L3 (p5/z)
    ///         whole chunk: 2 x (mova 4 vertical slices,  label 2L4
    ///           4 depth steps of A -> FMOPA)
    ///         partial chunk: 1 vertical slice per step   label 2L5
    ///     Trans (b_trans == 1):
    ///       pairs of depth steps: B rows (p6/z), A       label 2L8
    ///       one more step when d is odd                  label 2L7
    ///     epilogue of the block's mode                   labels 50 (dispatch), 51..54
    ///       for c in 0..ncols, tile t while 8t < m: slice c of za<t>, predicate p<t+1>
    ///   next column panel                                label 59 (66, 67: B panel step)
    /// next depth block: mode = mode_later, beta = one
    /// ```
    ///
    /// # Register map
    ///
    /// | Register | Use |
    /// |---|---|
    /// | `x1`, `x2` | `a`, `lda` in bytes |
    /// | `x3`, `x4`, `x5` | `b`, `ldb` in bytes, `b_trans` |
    /// | `x6`, `x7` | `c`, `ldc` in bytes |
    /// | `x8` | `ncols = min(8, n - j0)` |
    /// | `x9` | `C(0, j0)` |
    /// | `x10` | `op(B)(p0, j0)` |
    /// | `x11` | depth steps of the block not yet run |
    /// | `w12` | slice index: column of the panel (B loads, epilogue); `x12` also the 0 of `whilelt p5` |
    /// | `w13` | vertical slice index: depth step of the chunk; `x13` the C address in the epilogue |
    /// | `x14` | A at the current depth step; 8 (tile 1 row offset) in the epilogue |
    /// | `x15` | B of the chunk (NoTrans) or of the depth step (Trans); 16 in the epilogue |
    /// | `x16` | scratch: column address of a B load, double steps left, bit patterns; 24 in the epilogue |
    /// | `x20`, `x21`, `x22`, `x23` | `m`, `n`, `k`, `kc` |
    /// | `x24` | `p0` |
    /// | `x25` | mode of the current depth block (0 M0, 1 M1, 2 M2, 3 M3) |
    /// | `x26` | `d` |
    /// | `x27` | `A(0, p0)` |
    /// | `x28` | `j0` |
    /// | `z0`–`z3`, `z4`–`z7` | A vectors of two depth steps (tile `t` in `z{t}` / `z{4+t}`) |
    /// | `z8`–`z11` | B rows of up to four depth steps |
    /// | `z16`, `z17` | epilogue: the moved slice, the loaded C |
    /// | `z28`, `z29` | alpha, the block's beta (beta, then one) |
    /// | `p0` | all-true |
    /// | `p1`–`p4` | rows below `m` of tiles 0–3 |
    /// | `p5` | live depth steps of the chunk (NoTrans B column loads) |
    /// | `p6` | live columns of the panel (Trans B row loads) |
    /// | `pn9` | rows below `m` of tiles 0–3 (multi-vector A loads) |
    /// | `za0`–`za3` | the live C tiles; `za4` the transposer of NoTrans B (`za5`–`za7` unused) |
    ///
    /// Epilogue of tile `t`, column `c` (`pt` = `p<t+1>`, `addr` =
    /// `C(8t, j0 + c)`): M0 `st1d {za<t>h.d[w12, 0]}, pt, [addr]`; M1 MOVA,
    /// `ld1d` of C (`pt/z`), `fadd`, store; M2 MOVA, `fmul` by `z28`, store;
    /// M3 MOVA, `fmul`, `ld1d` of C, `fmla z16.d, pt/m, z17.d, z29.d`, store.
    ///
    /// It reads no A row `≥ m` and nothing of B outside `op(B)[0..k, 0..n]`,
    /// reads C only when the block's mode reads it, writes only
    /// `C[0..m, 0..n]`, never writes the argument block, and leaves through
    /// the frame's label 90.
    ///
    /// # Safety
    ///
    /// - `detect::thread_ready()` must have returned `true` on this thread
    ///   (SME2 with a 512-bit streaming vector length, not in streaming
    ///   mode, no live ZA state, no pending lazy save), and
    ///   `detect::enabled::<f64>()` must hold (FEAT_SME_F64F64).
    /// - `1 ≤ m ≤ 32`, `n ≥ 1`, `1 ≤ kc ≤ k`, `ldc ≥ m` (when `n > 1`),
    ///   `b_trans` 0 or 1, and the modes must be `Epilogue::select(alpha,
    ///   beta)` and `Epilogue::select(alpha, 1)`, `one_bits` the bits of 1.
    /// - Every address the layout of [`FewRowsArgs`] names for `i < m`,
    ///   `p < k`, `j < n` must be valid (C for reads and writes); C must not
    ///   overlap A, B or the argument block.
    pub(super) streaming fn sme2_few_rows_f64(args: &FewRowsArgs<f64>) {
        extensions: [f64f64],
        // @generated begin: tools/gen_sme2_few_rows.py, sme2_few_rows_f64 (edit the generator, not these lines)
        consts: [
            off_a = OFF_A,
            off_lda = OFF_LDA,
            off_b = OFF_B,
            off_ldb = OFF_LDB,
            off_b_trans = OFF_B_TRANS,
            off_c = OFF_C,
            off_ldc = OFF_LDC,
            off_m = OFF_M,
            off_n = OFF_N,
            off_k = OFF_K,
            off_kc = OFF_KC,
            off_alpha = OFF_ALPHA,
            off_beta = OFF_BETA,
            off_one = OFF_ONE,
            off_mode_first = OFF_MODE_FIRST,
            off_mode_later = OFF_MODE_LATER,
        ],
        prologue: [
            "ldr x1, [x0, #{off_a}]",
            "ldr x2, [x0, #{off_lda}]",
            "ldr x3, [x0, #{off_b}]",
            "ldr x4, [x0, #{off_ldb}]",
            "ldr x5, [x0, #{off_b_trans}]",
            "ldr x6, [x0, #{off_c}]",
            "ldr x7, [x0, #{off_ldc}]",
            "ldr x20, [x0, #{off_m}]",
            "ldr x21, [x0, #{off_n}]",
            "ldr x22, [x0, #{off_k}]",
            "ldr x23, [x0, #{off_kc}]",
            "ldr x25, [x0, #{off_mode_first}]",
            "lsl x2, x2, #3",
            "lsl x4, x4, #3",
            "lsl x7, x7, #3",
        ],
        body: [
            // p0 all-true; z28 alpha, z29 beta (the first depth block's).
            "ptrue p0.d",
            "ldr x16, [x0, #{off_alpha}]", "dup z28.d, x16",
            "ldr x16, [x0, #{off_beta}]", "dup z29.d, x16",
            // Rows below m: p1 (tile 0), pn9 (tiles 0-3, for the A loads), p2..p4 (tiles 1..).
            "mov x16, #0", "whilelt p1.d, x16, x20", "whilelt pn9.d, x16, x20, vlx4",
            "mov x16, #8", "whilelt p2.d, x16, x20",
            "mov x16, #16", "whilelt p3.d, x16, x20",
            "mov x16, #24", "whilelt p4.d, x16, x20",
            // x24 = p0 (first depth step of the block), x27 = A(0, p0).
            "mov x24, #0", "mov x27, x1",
            // Depth block: x26 = d = min(kc, k - p0); x28 = j0, x9 = C(0, j0),
            // x10 = B(p0, j0) of op(B): b + p0 (NoTrans, column-major) or b + p0 * ldb (Trans).
            "70:", "sub x26, x22, x24", "cmp x26, x23", "csel x26, x26, x23, lo",
            "mov x28, #0", "mov x9, x6", "cbnz x5, 71f", "add x10, x3, x24, lsl #3", "b 20f",
            "71:", "madd x10, x24, x4, x3",
            // Column panel: x8 = ncols = min(8, n - j0), p6 = columns below n; x14 = A,
            // x15 = B at depth p0 and column j0, x11 = depth steps left in the block.
            "20:", "sub x8, x21, x28", "mov x16, #8", "cmp x8, x16", "csel x8, x8, x16, lo",
            "whilelt p6.d, x28, x21", "mov x14, x27", "mov x15, x10", "mov x11, x26",
            // Live C tiles L = ceil(m / 8) select the variant 4(2L-1).
            "cmp x20, #24", "b.hi 47f",
            "cmp x20, #16", "b.hi 45f",
            "cmp x20, #8", "b.hi 43f",
            // 1 live C tile (1 to 8 rows): zero them, then the depth loop.
            "41:", "zero {{za0.d}}", "cbnz x5, 216f",
            // NoTrans B. Chunk of min(8, x11) depth steps: column c of the panel
            // (c < ncols) into horizontal slice c of za4.d, lanes of the live depth (p5).
            "212:", "mov w12, #0", "whilelt p5.d, x12, x11", "mov x16, x15",
            "213:", "ld1d {{za4h.d[w12, 0]}}, p5/z, [x16]", "add x16, x16, x4",
            "add w12, w12, #1", "cmp w12, w8", "b.lo 213b",
            // Vertical slice q of za4.d is the B row of depth step q of the chunk.
            "mov w13, #0", "cmp x11, #8", "b.lo 215f",
            // A whole chunk: four B rows per MOVA, 2 groups.
            "214:", "mova {{z8.d-z11.d}}, za4v.d[w13, 0:3]",
            "ld1d {{z0.d}}, p1/z, [x14]", "add x14, x14, x2", "ld1d {{z4.d}}, p1/z, [x14]",
            "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d",
            "fmopa za0.d, p0/m, p0/m, z9.d, z4.d",
            "ld1d {{z0.d}}, p1/z, [x14]", "add x14, x14, x2", "ld1d {{z4.d}}, p1/z, [x14]",
            "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z10.d, z0.d",
            "fmopa za0.d, p0/m, p0/m, z11.d, z4.d",
            "add w13, w13, #4", "cmp w13, #8", "b.lo 214b",
            "add x15, x15, #64", "subs x11, x11, #8", "b.ne 212b", "b 50f",
            // The last, partial chunk of the block: one depth step per B row.
            "215:", "mova z8.d, p0/m, za4v.d[w13, 0]",
            "ld1d {{z0.d}}, p1/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d",
            "add w13, w13, #1", "cmp w13, w11", "b.lo 215b", "b 50f",
            // Trans B: the B row of a depth step is contiguous; columns < n (p6). Two
            // steps per iteration, then one more when the block's depth is odd.
            "216:", "lsr x16, x11, #1", "cbz x16, 217f",
            "218:", "ld1d {{z8.d}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1d {{z0.d}}, p1/z, [x14]", "add x14, x14, x2",
            "ld1d {{z9.d}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1d {{z4.d}}, p1/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d",
            "fmopa za0.d, p0/m, p0/m, z9.d, z4.d",
            "subs x16, x16, #1", "b.ne 218b",
            "217:", "tbz x11, #0, 50f", "ld1d {{z8.d}}, p6/z, [x15]",
            "ld1d {{z0.d}}, p1/z, [x14]",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d",
            "b 50f",
            // 2 live C tiles (9 to 16 rows): zero them, then the depth loop.
            "43:", "zero {{za0.d, za1.d}}", "cbnz x5, 226f",
            // NoTrans B. Chunk of min(8, x11) depth steps: column c of the panel
            // (c < ncols) into horizontal slice c of za4.d, lanes of the live depth (p5).
            "222:", "mov w12, #0", "whilelt p5.d, x12, x11", "mov x16, x15",
            "223:", "ld1d {{za4h.d[w12, 0]}}, p5/z, [x16]", "add x16, x16, x4",
            "add w12, w12, #1", "cmp w12, w8", "b.lo 223b",
            // Vertical slice q of za4.d is the B row of depth step q of the chunk.
            "mov w13, #0", "cmp x11, #8", "b.lo 225f",
            // A whole chunk: four B rows per MOVA, 2 groups.
            "224:", "mova {{z8.d-z11.d}}, za4v.d[w13, 0:3]",
            "ld1d {{z0.d-z1.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z4.d-z5.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za0.d, p0/m, p0/m, z9.d, z4.d", "fmopa za1.d, p0/m, p0/m, z9.d, z5.d",
            "ld1d {{z0.d-z1.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z4.d-z5.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z10.d, z0.d", "fmopa za1.d, p0/m, p0/m, z10.d, z1.d",
            "fmopa za0.d, p0/m, p0/m, z11.d, z4.d", "fmopa za1.d, p0/m, p0/m, z11.d, z5.d",
            "add w13, w13, #4", "cmp w13, #8", "b.lo 224b",
            "add x15, x15, #64", "subs x11, x11, #8", "b.ne 222b", "b 50f",
            // The last, partial chunk of the block: one depth step per B row.
            "225:", "mova z8.d, p0/m, za4v.d[w13, 0]",
            "ld1d {{z0.d-z1.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "add w13, w13, #1", "cmp w13, w11", "b.lo 225b", "b 50f",
            // Trans B: the B row of a depth step is contiguous; columns < n (p6). Two
            // steps per iteration, then one more when the block's depth is odd.
            "226:", "lsr x16, x11, #1", "cbz x16, 227f",
            "228:", "ld1d {{z8.d}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1d {{z0.d-z1.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z9.d}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1d {{z4.d-z5.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za0.d, p0/m, p0/m, z9.d, z4.d", "fmopa za1.d, p0/m, p0/m, z9.d, z5.d",
            "subs x16, x16, #1", "b.ne 228b",
            "227:", "tbz x11, #0, 50f", "ld1d {{z8.d}}, p6/z, [x15]",
            "ld1d {{z0.d-z1.d}}, pn9/z, [x14]",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "b 50f",
            // 3 live C tiles (17 to 24 rows): zero them, then the depth loop.
            "45:", "zero {{za0.d, za1.d, za2.d}}", "cbnz x5, 236f",
            // NoTrans B. Chunk of min(8, x11) depth steps: column c of the panel
            // (c < ncols) into horizontal slice c of za4.d, lanes of the live depth (p5).
            "232:", "mov w12, #0", "whilelt p5.d, x12, x11", "mov x16, x15",
            "233:", "ld1d {{za4h.d[w12, 0]}}, p5/z, [x16]", "add x16, x16, x4",
            "add w12, w12, #1", "cmp w12, w8", "b.lo 233b",
            // Vertical slice q of za4.d is the B row of depth step q of the chunk.
            "mov w13, #0", "cmp x11, #8", "b.lo 235f",
            // A whole chunk: four B rows per MOVA, 2 groups.
            "234:", "mova {{z8.d-z11.d}}, za4v.d[w13, 0:3]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z4.d-z7.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z8.d, z2.d",
            "fmopa za0.d, p0/m, p0/m, z9.d, z4.d", "fmopa za1.d, p0/m, p0/m, z9.d, z5.d",
            "fmopa za2.d, p0/m, p0/m, z9.d, z6.d",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z4.d-z7.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z10.d, z0.d", "fmopa za1.d, p0/m, p0/m, z10.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z10.d, z2.d",
            "fmopa za0.d, p0/m, p0/m, z11.d, z4.d", "fmopa za1.d, p0/m, p0/m, z11.d, z5.d",
            "fmopa za2.d, p0/m, p0/m, z11.d, z6.d",
            "add w13, w13, #4", "cmp w13, #8", "b.lo 234b",
            "add x15, x15, #64", "subs x11, x11, #8", "b.ne 232b", "b 50f",
            // The last, partial chunk of the block: one depth step per B row.
            "235:", "mova z8.d, p0/m, za4v.d[w13, 0]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z8.d, z2.d",
            "add w13, w13, #1", "cmp w13, w11", "b.lo 235b", "b 50f",
            // Trans B: the B row of a depth step is contiguous; columns < n (p6). Two
            // steps per iteration, then one more when the block's depth is odd.
            "236:", "lsr x16, x11, #1", "cbz x16, 237f",
            "238:", "ld1d {{z8.d}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z9.d}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1d {{z4.d-z7.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z8.d, z2.d",
            "fmopa za0.d, p0/m, p0/m, z9.d, z4.d", "fmopa za1.d, p0/m, p0/m, z9.d, z5.d",
            "fmopa za2.d, p0/m, p0/m, z9.d, z6.d",
            "subs x16, x16, #1", "b.ne 238b",
            "237:", "tbz x11, #0, 50f", "ld1d {{z8.d}}, p6/z, [x15]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z8.d, z2.d",
            "b 50f",
            // 4 live C tiles (25 to 32 rows): zero them, then the depth loop.
            "47:", "zero {{za0.d, za1.d, za2.d, za3.d}}", "cbnz x5, 246f",
            // NoTrans B. Chunk of min(8, x11) depth steps: column c of the panel
            // (c < ncols) into horizontal slice c of za4.d, lanes of the live depth (p5).
            "242:", "mov w12, #0", "whilelt p5.d, x12, x11", "mov x16, x15",
            "243:", "ld1d {{za4h.d[w12, 0]}}, p5/z, [x16]", "add x16, x16, x4",
            "add w12, w12, #1", "cmp w12, w8", "b.lo 243b",
            // Vertical slice q of za4.d is the B row of depth step q of the chunk.
            "mov w13, #0", "cmp x11, #8", "b.lo 245f",
            // A whole chunk: four B rows per MOVA, 2 groups.
            "244:", "mova {{z8.d-z11.d}}, za4v.d[w13, 0:3]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z4.d-z7.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z8.d, z2.d", "fmopa za3.d, p0/m, p0/m, z8.d, z3.d",
            "fmopa za0.d, p0/m, p0/m, z9.d, z4.d", "fmopa za1.d, p0/m, p0/m, z9.d, z5.d",
            "fmopa za2.d, p0/m, p0/m, z9.d, z6.d", "fmopa za3.d, p0/m, p0/m, z9.d, z7.d",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z4.d-z7.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z10.d, z0.d", "fmopa za1.d, p0/m, p0/m, z10.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z10.d, z2.d", "fmopa za3.d, p0/m, p0/m, z10.d, z3.d",
            "fmopa za0.d, p0/m, p0/m, z11.d, z4.d", "fmopa za1.d, p0/m, p0/m, z11.d, z5.d",
            "fmopa za2.d, p0/m, p0/m, z11.d, z6.d", "fmopa za3.d, p0/m, p0/m, z11.d, z7.d",
            "add w13, w13, #4", "cmp w13, #8", "b.lo 244b",
            "add x15, x15, #64", "subs x11, x11, #8", "b.ne 242b", "b 50f",
            // The last, partial chunk of the block: one depth step per B row.
            "245:", "mova z8.d, p0/m, za4v.d[w13, 0]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z8.d, z2.d", "fmopa za3.d, p0/m, p0/m, z8.d, z3.d",
            "add w13, w13, #1", "cmp w13, w11", "b.lo 245b", "b 50f",
            // Trans B: the B row of a depth step is contiguous; columns < n (p6). Two
            // steps per iteration, then one more when the block's depth is odd.
            "246:", "lsr x16, x11, #1", "cbz x16, 247f",
            "248:", "ld1d {{z8.d}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "ld1d {{z9.d}}, p6/z, [x15]", "add x15, x15, x4",
            "ld1d {{z4.d-z7.d}}, pn9/z, [x14]", "add x14, x14, x2",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z8.d, z2.d", "fmopa za3.d, p0/m, p0/m, z8.d, z3.d",
            "fmopa za0.d, p0/m, p0/m, z9.d, z4.d", "fmopa za1.d, p0/m, p0/m, z9.d, z5.d",
            "fmopa za2.d, p0/m, p0/m, z9.d, z6.d", "fmopa za3.d, p0/m, p0/m, z9.d, z7.d",
            "subs x16, x16, #1", "b.ne 248b",
            "247:", "tbz x11, #0, 50f", "ld1d {{z8.d}}, p6/z, [x15]",
            "ld1d {{z0.d-z3.d}}, pn9/z, [x14]",
            "fmopa za0.d, p0/m, p0/m, z8.d, z0.d", "fmopa za1.d, p0/m, p0/m, z8.d, z1.d",
            "fmopa za2.d, p0/m, p0/m, z8.d, z2.d", "fmopa za3.d, p0/m, p0/m, z8.d, z3.d",
            "b 50f",
            // Epilogue of the 32 x 8 C tile of the column panel, by mode:
            // columns 0..ncols (w12), each tile while its first row is below m.
            "50:", "mov w12, #0", "mov x13, x9",
            "mov x14, #8", "mov x15, #16", "mov x16, #24",
            "cbz x25, 51f", "cmp x25, #2", "b.lo 52f", "b.eq 53f", "b 54f",
            // M0: C = acc. Direct ZA slice stores; C is not read.
            "51:",
            "st1d {{za0h.d[w12, 0]}}, p1, [x13]",
            "cmp x20, #8", "b.ls 55f",
            "st1d {{za1h.d[w12, 0]}}, p2, [x13, x14, lsl #3]",
            "cmp x20, #16", "b.ls 55f",
            "st1d {{za2h.d[w12, 0]}}, p3, [x13, x15, lsl #3]",
            "cmp x20, #24", "b.ls 55f",
            "st1d {{za3h.d[w12, 0]}}, p4, [x13, x16, lsl #3]",
            "55:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", "b.lo 51b",
            "b 59f",
            // M1: C = acc + C (SVE FADD, accumulator first).
            "52:",
            "mova z16.d, p0/m, za0h.d[w12, 0]", "ld1d {{z17.d}}, p1/z, [x13]",
            "fadd z16.d, p1/m, z16.d, z17.d", "st1d {{z16.d}}, p1, [x13]",
            "cmp x20, #8", "b.ls 56f",
            "mova z16.d, p0/m, za1h.d[w12, 0]", "ld1d {{z17.d}}, p2/z, [x13, x14, lsl #3]",
            "fadd z16.d, p2/m, z16.d, z17.d", "st1d {{z16.d}}, p2, [x13, x14, lsl #3]",
            "cmp x20, #16", "b.ls 56f",
            "mova z16.d, p0/m, za2h.d[w12, 0]", "ld1d {{z17.d}}, p3/z, [x13, x15, lsl #3]",
            "fadd z16.d, p3/m, z16.d, z17.d", "st1d {{z16.d}}, p3, [x13, x15, lsl #3]",
            "cmp x20, #24", "b.ls 56f",
            "mova z16.d, p0/m, za3h.d[w12, 0]", "ld1d {{z17.d}}, p4/z, [x13, x16, lsl #3]",
            "fadd z16.d, p4/m, z16.d, z17.d", "st1d {{z16.d}}, p4, [x13, x16, lsl #3]",
            "56:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", "b.lo 52b",
            "b 59f",
            // M2: C = acc * alpha (SVE FMUL); C is not read.
            "53:",
            "mova z16.d, p0/m, za0h.d[w12, 0]", "fmul z16.d, p1/m, z16.d, z28.d",
            "st1d {{z16.d}}, p1, [x13]",
            "cmp x20, #8", "b.ls 57f",
            "mova z16.d, p0/m, za1h.d[w12, 0]", "fmul z16.d, p2/m, z16.d, z28.d",
            "st1d {{z16.d}}, p2, [x13, x14, lsl #3]",
            "cmp x20, #16", "b.ls 57f",
            "mova z16.d, p0/m, za2h.d[w12, 0]", "fmul z16.d, p3/m, z16.d, z28.d",
            "st1d {{z16.d}}, p3, [x13, x15, lsl #3]",
            "cmp x20, #24", "b.ls 57f",
            "mova z16.d, p0/m, za3h.d[w12, 0]", "fmul z16.d, p4/m, z16.d, z28.d",
            "st1d {{z16.d}}, p4, [x13, x16, lsl #3]",
            "57:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", "b.lo 53b",
            "b 59f",
            // M3: t = acc * alpha (SVE FMUL), C = t + C * beta (SVE FMLA, one rounding).
            "54:",
            "mova z16.d, p0/m, za0h.d[w12, 0]", "fmul z16.d, p1/m, z16.d, z28.d",
            "ld1d {{z17.d}}, p1/z, [x13]", "fmla z16.d, p1/m, z17.d, z29.d",
            "st1d {{z16.d}}, p1, [x13]",
            "cmp x20, #8", "b.ls 58f",
            "mova z16.d, p0/m, za1h.d[w12, 0]", "fmul z16.d, p2/m, z16.d, z28.d",
            "ld1d {{z17.d}}, p2/z, [x13, x14, lsl #3]", "fmla z16.d, p2/m, z17.d, z29.d",
            "st1d {{z16.d}}, p2, [x13, x14, lsl #3]",
            "cmp x20, #16", "b.ls 58f",
            "mova z16.d, p0/m, za2h.d[w12, 0]", "fmul z16.d, p3/m, z16.d, z28.d",
            "ld1d {{z17.d}}, p3/z, [x13, x15, lsl #3]", "fmla z16.d, p3/m, z17.d, z29.d",
            "st1d {{z16.d}}, p3, [x13, x15, lsl #3]",
            "cmp x20, #24", "b.ls 58f",
            "mova z16.d, p0/m, za3h.d[w12, 0]", "fmul z16.d, p4/m, z16.d, z28.d",
            "ld1d {{z17.d}}, p4/z, [x13, x16, lsl #3]", "fmla z16.d, p4/m, z17.d, z29.d",
            "st1d {{z16.d}}, p4, [x13, x16, lsl #3]",
            "58:", "add x13, x13, x7", "add w12, w12, #1", "cmp w12, w8", "b.lo 54b",
            // Next column panel; then the next depth block, whose epilogue mode is the later
            // blocks' and whose beta is one. The end falls through to the exit label 90.
            "59:", "add x28, x28, #8", "add x9, x9, x7, lsl #3",
            "cbnz x5, 66f", "add x10, x10, x4, lsl #3", "b 67f",
            "66:", "add x10, x10, #64",
            "67:", "cmp x28, x21", "b.lo 20b",
            "add x24, x24, x23", "madd x27, x23, x2, x27",
            "ldr x25, [x0, #{off_mode_later}]",
            "ldr x16, [x0, #{off_one}]", "dup z29.d, x16",
            "cmp x24, x22", "b.lo 70b",
        ],
        // @generated end
    }
}
